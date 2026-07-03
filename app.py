import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_churn_model_pipeline.pkl")

model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html", form_data={})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": int(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"]),
        }

        df = pd.DataFrame([input_data])

        proba = model.predict_proba(df)
        # print("Prediction Probability:", proba, flush=True)

        prediction = model.predict(df)[0]
        probability = proba[0][1] * 100

        if probability >= 70:
            risk = "High Risk"
            recommendation = "Offer discount, improve support, and encourage long-term contract."
        elif probability >= 40:
            risk = "Medium Risk"
            recommendation = "Provide loyalty offer and monitor customer activity."
        else:
            risk = "Low Risk"
            recommendation = "Customer is likely to stay. Maintain service quality."

        result = "Customer is likely to Churn" if prediction == 1 else "Customer is likely to Stay"

        return render_template(
            "index.html",
            result=result,
            probability=round(probability, 2),
            risk=risk,
            recommendation=recommendation,
            form_data=input_data
        )

    except Exception as e:
        print("Error:", e, flush=True)
        return render_template("index.html", error=str(e), form_data=request.form)


if __name__ == "__main__":
    app.run(debug=False)