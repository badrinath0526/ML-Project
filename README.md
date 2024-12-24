# Diabetes Prediction Project

This project is an end-to-end machine learning pipeline for predicting diabetes based on medical and demographic attributes. The pipeline involves data preprocessing, handling class imbalance, training a Random Forest model, and deploying it as a web application.

---

## Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Web Application](#web-application)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Dataset
- **Source**: Kaggle ([Diabetes Prediction Dataset](https://www.kaggle.com/))
- **Samples**: 100,001
- **Features**: 9 (including the target variable `Diabetes`)

## Features
1. **Gender**
2. **Age**
3. **Hypertension**
4. **Heart Disease**
5. **Smoking History**
6. **BMI**
7. **HbA1c Level**
8. **Blood Glucose Level**
9. **Diabetes** (Target Variable: 1 = Diabetic, 0 = Non-diabetic)

---

## Workflow
1. **Data Cleaning and Preprocessing**:
   - Removed duplicates and handled missing values.
   - Encoded categorical variables and transformed skewed data (e.g., BMI, Blood Glucose).
   - Removed features with low importance (e.g., `gender`, `hypertension`).

2. **Class Imbalance Handling**:
   - Utilized KMeansSMOTE to generate synthetic samples for the minority class.

3. **Feature Engineering**:
   - Age values less than 1 were adjusted to ensure realistic inputs.
   - Log transformations applied to continuous features for stability.

4. **Model Training**:
   - Trained using RandomForestClassifier.
   - Hyperparameters tuned with GridSearchCV.

5. **Deployment**:
   - Model serialized with `joblib`.
   - Web application built using Flask for real-time predictions.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # For Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Web Application:
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open a browser and navigate to `http://127.0.0.1:5000`.

### API Endpoint:
- **POST `/predict`**:
   - Input: JSON with feature values.
   - Output: Predicted class (0 or 1).
   - Example:
     ```json
     {
         "age": 45,
         "bmi": 28.5,
         "blood_glucose_level": 150,
         "hba1c_level": 7.2,
         "smoking_history": "never"
     }
     ```

---

## Model Training and Evaluation
- **Model**: RandomForestClassifier
- **Metrics**:
  - Accuracy: 89%
  - Precision: 85%
  - Recall: 80%
  - F1-score: 82%
- **Feature Importance**: Top predictors include `age`, `bmi`, `hba1c_level`, and `blood_glucose_level`.

---

## Web Application
- **Backend**: Flask
- **Frontend**: HTML/CSS with Bootstrap
- **Deployment**: Dockerized and hosted on a cloud platform (e.g., AWS, Azure, Heroku).

---

## Results
The project successfully handles class imbalance, preprocesses data effectively, and predicts diabetes with high accuracy. The deployed web application enables users to interact with the model for real-time predictions.

---

## Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

---
