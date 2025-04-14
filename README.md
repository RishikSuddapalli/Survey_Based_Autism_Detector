# Autism Spectrum Disorder (ASD) Traits Prediction System

This project is a web application that predicts Autism Spectrum Disorder (ASD) traits based on user input using multiple machine learning models.

## Features

- **Multiple Model Predictions**: Utilizes 8 different machine learning models to provide comprehensive predictions
- **User-Friendly Interface**: Clean, responsive web forms for easy data input
- **Detailed Results**: Presents predictions from all models in an easy-to-understand format
- **Input Summary**: Displays a summary of the input data for verification

## Models Included

1. Logistic Regression
2. Decision Tree
3. AdaBoost
4. Gradient Boosting
5. Random Forest
6. SVM
7. XGBoost
8. Artificial Neural Network (ANN)

## Technologies Used

- **Backend**: FastAPI (Python)
- **Frontend**: HTML, CSS, Bootstrap
- **Templating**: Jinja2
- **Machine Learning**: Tensorflow XGBoost
- **Data Processing**: Pandas, NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/asd-prediction-system.git
   cd asd-prediction-system
   ```
 
2. Create a conda virtual environment and activate it:
   ```bash
   conda create -n survey_autism_detector python=3.11
   conda activate survey_autism_detector
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

5. Open your browser and navigate to:
   ```
   http://localhost:8000
   ```
![Screenshot 2025-04-07 112417](https://github.com/user-attachments/assets/7d246cc4-c57e-44d4-bf2f-9c4a9e2e27d0)

6. Fill the form.

![Screenshot 2025-04-07 112543](https://github.com/user-attachments/assets/6f3006bd-c37a-4ac9-9245-6d0cf0a45865)

7.Click on "Predict ASD Traits"

![Screenshot 2025-04-14 212603](https://github.com/user-attachments/assets/68facf82-094b-4bd2-b4d4-0fb532fc2608)

## Project Structure

```
asd-prediction-system/
├── main.py                # FastAPI application and routes
├── models/                # Directory containing trained models
├── static/                # Static files (CSS, JS, images)
├── templates/             # HTML templates
│   ├── form.html          # Input form template
│   └── result.html        # Results display template
├── scaler.joblib          # Scaler object for feature normalization
├── feature_names.json     # List of feature names for model input
├── requirements.txt       # Python dependencies
├── Autism-detection.ipynb # for Data Analysis and Model Training
├── autism_screening.csv
├── data_csv.csv
├── Toddler Autism dataset July 2018.csv  
└── README.md              # This file
```




## Usage

1. Fill out the form with the required information:
   - Basic demographic information
   - Behavioral questions (answered with Yes/No)
   
2. Click "Predict ASD Traits" to submit the form

3. View the predictions from all models and verify your input data

## Requirements

The `requirements.txt` file should include:

```
fastapi
uvicorn
joblib
numpy
pandas
scikit-learn
jinja2
python-multipart
```

## Acknowledgments

- The Autism Spectrum Disorder screening dataset
- FastAPI and Uvicorn for the web framework
- Tensorfloq for machine learning tools
- Bootstrap for frontend components
