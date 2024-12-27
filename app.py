from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("model1.pkl")
regressor=joblib.load("regressor.pkl")
scaler=joblib.load("scaler_regression.pkl")

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

            scaled_input_data=scaler.transform(input_data)
            y_pred=regressor.predict(scaled_input_data)[0]
            print(y_pred)
            y_pred=np.round(y_pred*100,2)


            # Make prediction
            prediction = model.predict(input_data)
            print(prediction)


            # prediction_prob = model.predict_proba(input_data)
        
            # print(prediction_prob)
        
            # if prediction_prob[0][0] > prediction_prob[0][1]:
            #     prediction_text = "Your diabetes result is negative with {:.2f}% accuracy".format(prediction_prob[0][0] * 100)
            # else:
            #     prediction_text = "Your diabetes result is positive with {:.2f}% accuracy".format(prediction_prob[0][1] * 100)


 

            # Return prediction result
            if prediction[0] == 0:
                prediction_text = " Negative for diabetes, but there is a "+str(y_pred)+"% chance of being positive. Please consult a healthcare professional for further evaluation."
            else:
                prediction_text = "Positive for diabetes ,but there is a  "+str(100-y_pred)+"% chance of being negative. Please consult a healthcare professional for confirmation and guidance."
            
            # Render the template with prediction
            return render_template("index.html", prediction_text=prediction_text)

        except Exception as e:
            return render_template("index.html", prediction_text="Error: " + str(e))

    else:
        # Handle the GET request (initial page load)
        return render_template("index.html", prediction_text="")

if __name__ == "__main__":
    app.run(debug=True)
