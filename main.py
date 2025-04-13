from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
import pandas as pd
import json
from pydantic import BaseModel
from typing import List, Dict
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load all models
models = {
    "Logistic Regression": joblib.load('logreg_model.joblib'),
    "Decision Tree": joblib.load('DTC_model.joblib'),
    "AdaBoost": joblib.load('ABC_model.joblib'),
    "Gradient Boosting": joblib.load('GBC_model.joblib'),
    "Random Forest": joblib.load('RFC_model.joblib'),
    "SVM": joblib.load('SVC_model.joblib'),
    "XGBoost": joblib.load('xgboost_model.joblib')
}

# Load scaler and feature names
scaler = joblib.load('scaler.joblib')
with open('feature_names.json', 'r') as f:
    FEATURE_NAMES = json.load(f)

# Initialize label encoders
le = LabelEncoder()

def preprocess_input(data: dict):
    # Create DataFrame from input
    input_df = pd.DataFrame([data])
    
    # Label encode categorical variables
    input_df['Sex'] = le.fit_transform(input_df['Sex'])
    input_df['Jaundice'] = le.fit_transform(input_df['Jaundice'])
    input_df['Family_mem_with_ASD'] = le.fit_transform(input_df['Family_mem_with_ASD'])
    
    # One-hot encoding with all possible categories
    ethnicities = ['Asian', 'Black', 'Hispanic', 'Latino', 'Middle Eastern', 
                  'Mixed', 'Native Indian', 'Others', 'Pacifica', 'South Asian', 
                  'Turkish', 'White European', 'White-European', 'others']
    
    completers = ['Family Member', 'Health Care Professional', 'Others', 
                 'Parent', 'Relative', 'School and NGO', 'Self']
    
    # Create all possible columns with 0 values
    for eth in ethnicities:
        input_df[f'Ethnicity_{eth}'] = 0
    for comp in completers:
        input_df[f'Who_completed_the_test_{comp}'] = 0
    
    # Set the actual selected values to 1
    selected_ethnicity = input_df['Ethnicity'].iloc[0]
    selected_completer = input_df['Who_completed_the_test'].iloc[0]
    
    # Handle unknown categories
    if f'Ethnicity_{selected_ethnicity}' not in input_df.columns:
        selected_ethnicity = 'Others'
    if f'Who_completed_the_test_{selected_completer}' not in input_df.columns:
        selected_completer = 'Others'
    
    input_df[f'Ethnicity_{selected_ethnicity}'] = 1
    input_df[f'Who_completed_the_test_{selected_completer}'] = 1
    
    # Drop original columns
    input_df = input_df.drop(['Ethnicity', 'Who_completed_the_test'], axis=1)
    
    # Ensure we have all expected features
    for feature in FEATURE_NAMES:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Reorder columns to match training data
    input_df = input_df[FEATURE_NAMES]
    
    # Scale the data
    scaled_input = scaler.transform(input_df)
    
    return scaled_input

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                 A1: int = Form(...),
                 A2: int = Form(...),
                 A3: int = Form(...),
                 A4: int = Form(...),
                 A5: int = Form(...),
                 A6: int = Form(...),
                 A7: int = Form(...),
                 A8: int = Form(...),
                 A9: int = Form(...),
                 A10: int = Form(...),
                 Age_Years: int = Form(...),
                 Qchat_10_Score: float = Form(...),
                 Sex: str = Form(...),
                 Ethnicity: str = Form(...),
                 Jaundice: str = Form(...),
                 Family_mem_with_ASD: str = Form(...),
                 Who_completed_the_test: str = Form(...)):
    
    input_data = {
        'A1': A1,
        'A2': A2,
        'A3': A3,
        'A4': A4,
        'A5': A5,
        'A6': A6,
        'A7': A7,
        'A8': A8,
        'A9': A9,
        'A10_Autism_Spectrum_Quotient': A10,
        'Age_Years': Age_Years,
        'Qchat_10_Score': Qchat_10_Score,
        'Sex': Sex,
        'Ethnicity': Ethnicity,
        'Jaundice': Jaundice,
        'Family_mem_with_ASD': Family_mem_with_ASD,
        'Who_completed_the_test': Who_completed_the_test
    }
    
    # Preprocess input
    processed_input = preprocess_input(input_data)
    
    # Get predictions from all models
    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(processed_input)
        predictions[model_name] = 'Yes' if pred[0] == 1 else 'No'
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "predictions": predictions,
        "input_data": input_data
    })