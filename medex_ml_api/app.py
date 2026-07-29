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

# Joblib pickles models with varying import paths, so expose the class under both __main__ and ml_models
import types
sys.modules["__main__"].ThresholdClassifier = ThresholdClassifier

ml_models_mod = types.ModuleType("ml_models")
training_mod = types.ModuleType("ml_models.training")
train_models_mod = types.ModuleType("ml_models.training.train_models")
train_models_mod.ThresholdClassifier = ThresholdClassifier
training_mod.train_models = train_models_mod
ml_models_mod.training = training_mod
sys.modules["ml_models"] = ml_models_mod
sys.modules["ml_models.training"] = training_mod
sys.modules["ml_models.training.train_models"] = train_models_mod

# ======================
# APP CONFIG
# ======================

app = Flask(__name__)
CORS(app)


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "404 Not Found",
        "path": request.path,
        "path_info": request.environ.get("PATH_INFO"),
        "script_name": request.environ.get("SCRIPT_NAME"),
        "request_uri": request.environ.get("REQUEST_URI"),
        "raw_path": request.environ.get("RAW_URI")
    }), 404


class VercelPathFixer:

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        raw_uri = environ.get("REQUEST_URI", "") or environ.get("RAW_URI", "")
        if raw_uri:
            path = raw_uri.split("?")[0]
        else:
            path = environ.get("PATH_INFO", "")

        for prefix in ["/api/index.py", "/api/index", "/api"]:
            if path.startswith(prefix):
                path = path[len(prefix):]
                break

        if not path or not path.startswith("/"):
            path = "/" + path.lstrip("/")

        environ["PATH_INFO"] = path
        return self.wsgi_app(environ, start_response)

app.wsgi_app = VercelPathFixer(app.wsgi_app)




# ======================
# LOAD MODELS
# ======================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
if not MODELS_DIR.exists():
    # Fallback for Vercel lambda filesystem layout
    MODELS_DIR = Path("/var/task/medex_ml_api/models")

print(f"Loading models from: {MODELS_DIR}")

try:
    heart_model = joblib.load(MODELS_DIR / "heart_model.joblib")
    diabetes_model = joblib.load(MODELS_DIR / "diabetes_model.joblib")
    bp_model = joblib.load(MODELS_DIR / "bp_model.joblib")
    print("All ML models loaded successfully!")
except Exception as err:
    print(f"Error loading ML models: {err}")
    heart_model = None
    diabetes_model = None
    bp_model = None



def build_response(
    risk,
    confidence,
    title,
    description,
    tips
):

    return {
        "success": True,
        "risk": risk,
        "confidence": round(confidence, 2),
        "title": title,
        "description": description,
        "tips": list(set(tips)),
        "disclaimer": (
            "This prediction is informational only and not a medical diagnosis."
        )
    }


def derive_binary_risk(
    probability,
    high_threshold,
    moderate_threshold
):

    if probability >= high_threshold:
        return "High Risk"

    if probability >= moderate_threshold:
        return "Moderate Risk"

    return "Low Risk"


# ======================
# HEART EXPLANATION
# ======================

def explain_heart(
    data,
    risk,
    confidence
):

    reasons = []
    tips = []

    # AGE
    if data["age"] >= 55:
        reasons.append(
            "age-related cardiovascular risk"
        )

    if data["blood_pressure"] >= 140:
        reasons.append(
            "high blood pressure"
        )
        tips.append(
            "Reduce salty foods"
        )

    if data["cholesterol"] >= 240:
        reasons.append(
            "high cholesterol"
        )
        tips.append(
            "Reduce oily foods"
        )

    # HEART RATE
    if data["heart_rate"] > 100:
        reasons.append(
            "elevated heart rate"
        )
        tips.append(
            "Monitor heart rate"
        )

    elif data["heart_rate"] < 60:
        reasons.append(
            "low resting heart rate"
        )

    # SMOKING
    if data["smoking"] == 1:
        reasons.append(
            "smoking habit"
        )
        tips.append(
            "Avoid smoking"
        )

    # DIABETES
    if data["diabetes"] == 1:
        reasons.append(
            "diabetes-related risk"
        )
        tips.append(
            "Monitor blood sugar"
        )

    # CHEST PAIN
    if (
        data[
            "exercise_chest_pain"
        ] == 1
    ):
        reasons.append(
            "exercise-related chest pain"
        )

    # CHEST PAIN TYPE
    if (
        data[
            "chest_pain_type"
        ] >= 2
    ):
        reasons.append(
            "chest pain indicators"
        )

    # DEFAULT HEALTH TIPS
    tips.extend([
        "Stay physically active",
        "Maintain balanced nutrition",
        "Get regular health checkups"
    ])

    title = (
        "High Heart Risk Detected"
        if risk == "High Risk"
        else "Moderate Heart Risk"
        if risk == "Moderate Risk"
        else "Low Heart Risk"
    )

    if reasons:
        description = (
            "Your result may be influenced by "
            + ", ".join(reasons)
            + "."
        )
    else:
        if risk == "High Risk":

            description = (
                "Your prediction suggests increased heart-related risk patterns even though strong visible risk factors were limited."
            )

        elif risk == (
            "Moderate Risk"
        ):

            description = (
                "Some indicators suggest a moderate heart-related risk."
            )

        else:

            description = (
                "Your current inputs suggest a lower heart-related risk."
            )

    return build_response(
        risk,
        confidence,
        title,
        description,
        tips
    )


# ======================
# DIABETES EXPLANATION
# ======================

def explain_diabetes(
    data,
    risk,
    confidence
):

    reasons = []
    tips = []

    if data["glucose"] >= 180:
        reasons.append(
            "high glucose levels"
        )
        tips.append(
            "Reduce sugary foods"
        )

    if data["blood_pressure"] >= 90:
        reasons.append(
            "elevated blood pressure"
        )

    if data["weight"] >= 85:
        reasons.append(
            "higher body weight"
        )
        tips.append(
            "Maintain healthy weight"
        )

    if (
        data[
            "family_history"
        ]
    ):
        reasons.append(
            "family history of diabetes"
        )

    if data["age"] > 50:
        reasons.append(
            "age-related risk"
        )

    tips.extend([
        "Exercise regularly",
        "Stay hydrated"
    ])

    title = (
        "High Diabetes Risk"
        if risk == "High Risk"
        else "Moderate Diabetes Risk"
        if risk == "Moderate Risk"
        else "Low Diabetes Risk"
    )

    if reasons:
        description = (
            "Your result may be influenced by "
            + ", ".join(reasons)
            + "."
        )
    else:
        description = (
            "Your current inputs show lower diabetes risk indicators."
        )

    return build_response(
        risk,
        confidence,
        title,
        description,
        tips
    )


# ======================
# BP EXPLANATION
# ======================

def explain_bp(
    data,
    risk,
    confidence
):

    reasons = []
    tips = []

    if (
        data["systolic_bp"]
        >= 140
        or
        data["diastolic_bp"]
        >= 90
    ):
        reasons.append(
            "high blood pressure"
        )
        tips.append(
            "Reduce salt intake"
        )

    if data["stress_level"] >= 7:
        reasons.append(
            "high stress levels"
        )
        tips.append(
            "Practice relaxation"
        )

    if data["sleep_hours"] < 6:
        reasons.append(
            "poor sleep quality"
        )
        tips.append(
            "Improve sleep routine"
        )

    if data["smoking"] == 1:
        reasons.append(
            "smoking habit"
        )
        tips.append(
            "Avoid smoking"
        )

    if (
        data[
            "physical_activity"
        ] < 2
    ):
        reasons.append(
            "low physical activity"
        )
        tips.append(
            "Exercise daily"
        )

    title = (
        "High Blood Pressure Risk"
        if risk == "High Risk"
        else (
            "Elevated Blood Pressure"
            if risk == "Elevated"
            else "Healthy Blood Pressure"
        )
    )

    if reasons:
        description = (
            "Your result may be influenced by "
            + ", ".join(reasons)
            + "."
        )
    else:
        description = (
            "Your blood pressure indicators look healthy."
        )

    return build_response(
        risk,
        confidence,
        title,
        description,
        tips
    )

# ======================
# HEALTH CHECK
# ======================

@app.route("/debug-path", defaults={"path": ""})
@app.route("/debug-path/<path:path>")
def debug_path(path):
    env_subset = {
        k: str(v) for k, v in request.environ.items()
        if any(x in k.upper() for x in ["PATH", "URI", "URL", "HTTP_", "VERCEL"])
    }
    return jsonify({
        "received_path_param": path,
        "request_path": request.path,
        "env_subset": env_subset
    })



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
        probability = float(heart_model.predict_proba(df)[0][1])
        confidence = probability * 100
        risk = derive_binary_risk(
            probability,
            high_threshold=0.75,
            moderate_threshold=0.40
        )

        return jsonify(
            explain_heart(
                data,
                risk,
                confidence
            )
        )

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
        probability = float(diabetes_model.predict_proba(df)[0][1])
        confidence = probability * 100
        risk = derive_binary_risk(
            probability,
            high_threshold=0.75,
            moderate_threshold=0.45
        )

        return jsonify(
            explain_diabetes(
                data,
                risk,
                confidence
            )
        )

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

        probability = float(bp_model.predict_proba(df)[0][1])
        confidence = probability * 100

        if probability >= 0.75:
            risk = "High Risk"
        elif probability >= 0.40:
            risk = "Elevated"
        else:
            risk = "Normal"

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
                if risk != "High Risk":
                    risk = "Elevated"

        return jsonify(
            explain_bp(
                data,
                risk,
                confidence
            )
        )

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