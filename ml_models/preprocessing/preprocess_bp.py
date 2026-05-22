import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# LOAD DATASET
df = pd.read_csv(
    "datasets/bp.csv"
)

print("\nOriginal Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# REMOVE DUPLICATES
df.drop_duplicates(inplace=True)

# HANDLE MISSING VALUES
for col in df.columns:

    if df[col].dtype in [
        'int64',
        'float64'
    ]:
        df[col].fillna(
            df[col].median(),
            inplace=True
        )

    else:
        df[col].fillna(
            df[col].mode()[0],
            inplace=True
        )

# LABEL ENCODING
encoder = LabelEncoder()

categorical_cols = df.select_dtypes(
    include=['object']
).columns

for col in categorical_cols:
    df[col] = encoder.fit_transform(
        df[col]
    )

# REMOVE OUTLIERS
numeric_cols = df.select_dtypes(
    include=np.number
).columns

for col in numeric_cols:

    if col != "Risk":

        Q1 = df[col].quantile(0.25)

        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - (
            1.5 * IQR
        )

        upper = Q3 + (
            1.5 * IQR
        )

        df = df[
            (df[col] >= lower)
            &
            (df[col] <= upper)
        ]

print("\nCleaned Shape:")
print(df.shape)

df.to_csv(
    "processed/bp_cleaned.csv",
    index=False
)

print(
    "\nBlood pressure dataset cleaned successfully!"
)

# ==========================
# FEATURE SELECTION
# ==========================

output_path = Path("processed")
output_path.mkdir(parents=True, exist_ok=True)

bp_desired = [
    "age",
    "gender",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "smoking",
    "stress_level",
    "sleep_hours",
    "physical_activity",
]

# mapping from desired name -> existing column name in cleaned df
bp_map = {
    "gender": "male",
    "bmi": "BMI",
    "systolic_bp": "sysBP",
    "diastolic_bp": "diaBP",
    "heart_rate": "heartRate",
    "smoking": "currentSmoker",
}

df_features = pd.DataFrame()

for name in bp_desired:
    src = bp_map.get(name, name)
    if src in df.columns:
        df_features[name] = df[src]
    else:
        df_features[name] = np.nan

# target mapping: preserve the original binary `Risk` labels as `bp_risk`
if "Risk" in df.columns:
    df_features["bp_risk"] = df["Risk"].astype(int)
elif "bp_risk" in df.columns:
    df_features["bp_risk"] = df["bp_risk"].astype(int)
else:
    df_features["bp_risk"] = np.nan

df_features.to_csv(
    output_path / "bp_features.csv",
    index=False
)

print("\nSaved feature-selected file:")
print(output_path / "bp_features.csv")