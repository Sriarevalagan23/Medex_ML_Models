Medex ML API

Quick start

Prerequisites:
- Python 3.8+ (project uses a virtual environment at `../.venv`) 

Install dependencies (from repo root):

```bash
# create venv if you don't have one
python3 -m venv .venv
source .venv/bin/activate
pip install -r medex_ml_api/requirements.txt
```

Run the API (from repo root):

```bash
# runs on PORT 5001 by default in these instructions
cd medex_ml_api
PORT=5001 ../.venv/bin/python app.py
```

Health check

```bash
curl http://127.0.0.1:5000/
```

Example requests

- Predict heart risk

```bash
curl -X POST http://127.0.0.1:5000/predict-heart \
  -H "Content-Type: application/json" \
  -d '{"age":55, "gender":1, "chest_pain_type":2, "blood_pressure":140, "cholesterol":240, "heart_rate":150, "exercise_chest_pain":0, "diabetes":0, "smoking":0}'
```

- Predict diabetes risk

```bash
curl -X POST http://127.0.0.1:5000/predict-diabetes \
  -H "Content-Type: application/json" \
  -d '{"age":45, "glucose":130, "blood_pressure":80, "height":165, "weight":70, "family_history":false}'
```

- Predict blood pressure risk

```bash
curl -X POST http://127.0.0.1:5000/predict-bp \
  -H "Content-Type: application/json" \
  -d '{"age":50, "gender":1, "height":170, "weight":75, "systolic_bp":135, "diastolic_bp":85, "heart_rate":72, "smoking":0, "stress_level":3, "sleep_hours":7, "physical_activity":2}'
```

Notes
- Models must be located in `medex_ml_api/models/` and named `heart_model.joblib`, `diabetes_model.joblib`, and `bp_model.joblib` (those files currently exist in the project).
- A small `ThresholdClassifier` compatibility wrapper was added to `app.py` to allow unpickling a threshold-wrapped estimator used during training.
- This is a development server; use a WSGI server (gunicorn/uvicorn) for production.
