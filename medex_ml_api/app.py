from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import sys
from pathlib import Path


# Try to import the ThresholdClassifier used during training so joblib can
# successfully unpickle models that reference it. Fall back to a local
# compatible implementation if import fails.
try:
    from ml_models.training.train_models import ThresholdClassifier
except Exception:
    from sklearn.base import BaseEstimator, ClassifierMixin

    class ThresholdClassifier(BaseEstimator, ClassifierMixin):
        def __init__(self, estimator=None, threshold=0.5):
            self.estimator = estimator
            self.threshold = threshold

        def fit(self, X, y):
            self.estimator.fit(X, y)
            self.is_fitted_ = True
            return self

        def predict_proba(self, X):
            return self.estimator.predict_proba(X)

        def predict(self, X):
            probabilities = self.predict_proba(X)[:, 1]
            return (probabilities >= self.threshold).astype(int)

# Joblib pickles the diabetes model with `__main__.ThresholdClassifier` as the
# import path, so expose the class under that module name for Gunicorn.
sys.modules["__main__"].ThresholdClassifier = ThresholdClassifier

# ======================
# APP CONFIG
# ======================

app = Flask(__name__)
CORS(app)

# ======================
# LOAD MODELS
# ======================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

heart_model = joblib.load(
    MODELS_DIR / "heart_model.joblib"
)

diabetes_model = joblib.load(
    MODELS_DIR / "diabetes_model.joblib"
)

bp_model = joblib.load(
    MODELS_DIR / "bp_model.joblib"
)

# ======================
# HEALTH CHECK
# ======================

@app.route("/")
def home():
    return jsonify({
        "success": True,
        "message":
        "Medex ML API Running"
    })


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy"
    })


# ======================
# HEART PREDICTION
# ======================

@app.route(
    "/predict-heart",
    methods=["POST"]
)
def predict_heart():

    try:

        data = request.json

        age = data["age"]
        sex = data["gender"]
        cp = data["chest_pain_type"]
        trestbps = data["blood_pressure"]
        chol = data["cholesterol"]
        thalach = data["heart_rate"]
        exang = data["exercise_chest_pain"]

        diabetes = data["diabetes"]
        smoking = data["smoking"]

        # Default values
        fbs = diabetes
        restecg = 0
        oldpeak = 1.0
        slope = 1
        ca = 0
        thal = 1

        # Build a DataFrame with the same column names used during training
        df = pd.DataFrame([
            {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalch": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }
        ])

        prediction = int(heart_model.predict(df)[0])
        confidence = float(heart_model.predict_proba(df)[0][1])

        return jsonify({
            "success": True,
            "risk":
            "High Risk"
            if prediction == 1
            else "Low Risk",

            "confidence":
            round(
                confidence * 100,
                2
            )
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ======================
# DIABETES PREDICTION
# ======================

@app.route(
    "/predict-diabetes",
    methods=["POST"]
)
def predict_diabetes():

    try:

        data = request.json

        age = data["age"]
        glucose = data["glucose"]
        blood_pressure = data["blood_pressure"]

        height = data["height"]
        weight = data["weight"]

        pregnancies = data.get(
            "pregnancies",
            0
        )

        family_history = (
            data[
                "family_history"
            ]
        )

        # BMI calculation
        height_m = height / 100

        bmi = (
            weight /
            (
                height_m ** 2
            )
        )

        insulin = 85

        pedigree = (
            1
            if family_history
            else 0
        )

        # The diabetes training data uses columns: Pregnancies, Glucose, BMI, Age, DiabetesPedigreeFunction
        df = pd.DataFrame([
            {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BMI": bmi,
                "Age": age,
                "DiabetesPedigreeFunction": pedigree,
            }
        ])

        prediction = int(diabetes_model.predict(df)[0])
        confidence = float(diabetes_model.predict_proba(df)[0][1])

        return jsonify({
            "success": True,
            "risk":
            "High Risk"
            if prediction == 1
            else "Low Risk",

            "confidence":
            round(
                confidence * 100,
                2
            )
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ======================
# BP PREDICTION
# ======================

@app.route(
    "/predict-bp",
    methods=["POST"]
)
def predict_bp():

    try:

        data = request.json

        height = data["height"]

        weight = data["weight"]

        bmi = (
            weight /
            (
                (
                    height / 100
                ) ** 2
            )
        )

        df = pd.DataFrame([
            {
                "age": data["age"],
                "gender": data["gender"],
                "bmi": bmi,
                "systolic_bp": data["systolic_bp"],
                "diastolic_bp": data["diastolic_bp"],
                "heart_rate": data["heart_rate"],
                "smoking": data["smoking"],
                "stress_level": data.get("stress_level", np.nan),
                "sleep_hours": data.get("sleep_hours", np.nan),
                "physical_activity": data.get("physical_activity", np.nan),
            }
        ])

        prediction = int(bp_model.predict(df)[0])
        confidence = float(bp_model.predict_proba(df)[0][1])

        risk_levels = {
            0: "Normal",
            1: "Elevated",
            2: "High Risk",
        }

        # Start with model's prediction
        risk = risk_levels.get(prediction, "Unknown")

        # Medical safeguard override: if measured BP crosses clinical thresholds,
        # override the model output to be medically conservative.
        try:
            systolic = float(data.get("systolic_bp", 0))
            diastolic = float(data.get("diastolic_bp", 0))
        except Exception:
            systolic = None
            diastolic = None

        if systolic is not None and diastolic is not None:
            if (systolic >= 140) or (diastolic >= 90):
                risk = "High Risk"
            elif (systolic >= 120) or (diastolic >= 80):
                # Only escalate to Elevated if model didn't already predict High
                if risk != "High Risk":
                    risk = "Elevated"

        return jsonify({
            "success": True,
            "risk": risk,
            "confidence": round(confidence * 100, 2),
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
# ======================
# RUN SERVER
# ======================

if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", "5000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )