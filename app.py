from flask import Flask,render_template,request
import joblib
import numpy as np


app=Flask(__name__)
model=joblib.load("model.pkl")
le_gender=joblib.load("le_gender.pkl")
le_smoking_history=joblib.load("le_smoking_history.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_diabetes",methods=['POST','GET'])
def predict_diabetes():
    if request.method=='POST':
        age=float(request.form['age'])
        gender=request.form['gender']
        hypertension=request.form['hypertension']
        heart_disease=request.form['heart_disease']
        smoking_history=request.form['smoking_history']
        bmi=float(request.form['bmi'])
        HbA1c=float(request.form['HbA1c'])
        blood_glucose=float(request.form['blood_glucose'])
    
        gender_encoded=le_gender.transform([gender])[0]
        smoking_history_encoded=le_smoking_history.transform([smoking_history])[0]

        input_data=np.array([age,gender_encoded,hypertension,heart_disease,smoking_history_encoded,bmi,HbA1c,blood_glucose]).reshape(1,-1)
        
        

        prediction=model.predict(input_data)
        print(prediction)
        if(prediction[0]==0):
           prediction_text="Your diabetes result is negative"
        else:
            prediction_text="Your diabetes result is positive"
       
        return render_template("index.html",prediction_text=prediction_text)

if __name__=="__main__":
    app.run(debug=True)

