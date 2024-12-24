from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("model1.pkl")
# le_gender = joblib.load("le_gender.pkl")
# le_smoking_history = joblib.load("le_smoking_history.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_diabetes", methods=['POST', 'GET'])
def predict_diabetes():
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            bmi = float(request.form['bmi'])
            HbA1c = float(request.form['HbA1c'])
            blood_glucose = float(request.form['blood_glucose'])

            # Prepare input data as 2D array (one row, four columns)
            input_data = pd.DataFrame([[age, bmi, HbA1c, blood_glucose]])

            # Make prediction
            prediction = model.predict(input_data)
            print(prediction)

            # Return prediction result
            if prediction[0] == 0:
                prediction_text = "Your diabetes result is negative"
            else:
                prediction_text = "Your diabetes result is positive"
            
            # Render the template with prediction
            return render_template("index.html", prediction_text=prediction_text)

        except Exception as e:
            return render_template("index.html", prediction_text="Error: " + str(e))

    else:
        # Handle the GET request (initial page load)
        return render_template("index.html", prediction_text="")

if __name__ == "__main__":
    app.run(debug=True)
