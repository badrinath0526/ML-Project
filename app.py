from flask import Flask,render_template,request

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def home():
    if request.method=='POST':
        age=request.form['age']
        gender=request.form['gender']
        hypertension=request.form['hypertension']
        heart_disease=request.form['heart_disease']
        bmi=request.form['bmi']
        HbA1c=request.form['HbA1c']
        blood_glucose=request.form['blood_glucose']

        return render_template("result.html",age=age,gender=gender,hypertension=hypertension,heart_disease=heart_disease,bmi=bmi,HbA1c=HbA1c,blood_glucose=blood_glucose)
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)

