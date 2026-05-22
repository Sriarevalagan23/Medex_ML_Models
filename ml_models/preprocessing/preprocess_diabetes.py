import pandas as pd
import numpy as np
from pathlib import Path

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv(
    "datasets/diabetes.csv"
)

print("\nOriginal Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values Before:")
print(df.isnull().sum())


# ==========================
# REMOVE DUPLICATES
# ==========================

df.drop_duplicates(
    inplace=True
)


# ==========================
# REPLACE INVALID ZEROS
# (0 means missing in these)
# ==========================

columns_with_invalid_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in columns_with_invalid_zero:
    df[col] = df[col].replace(
        0,
        np.nan
    )


# ==========================
# HANDLE MISSING VALUES
# Use median for robustness
# ==========================

for col in df.columns:

    if col != "Outcome":

        df[col].fillna(
            df[col].median(),
            inplace=True
        )


# ==========================
# REMOVE OUTLIERS (IQR)
# ==========================

numeric_columns = df.select_dtypes(
    include=np.number
).columns

for col in numeric_columns:

    if col != "Outcome":

        Q1 = df[col].quantile(
            0.25
        )

        Q3 = df[col].quantile(
            0.75
        )

        IQR = Q3 - Q1

        lower_bound = (
            Q1 -
            1.5 * IQR
        )

        upper_bound = (
            Q3 +
            1.5 * IQR
        )

        df = df[
            (
                df[col]
                >= lower_bound
            )
            &
            (
                df[col]
                <= upper_bound
            )
        ]


# ==========================
# FINAL CHECKS
# ==========================

print(
    "\nCleaned Shape:"
)
print(df.shape)

print(
    "\nMissing Values After:"
)
print(
    df.isnull().sum()
)

print(
    "\nTarget Distribution:"
)
print(
    df["Outcome"]
    .value_counts()
)

print(
    "\nData Types:"
)
print(
    df.dtypes
)

print(
    "\nDataset Summary:"
)
print(
    df.describe()
)


# ==========================
# SAVE CLEAN DATA
# ==========================

df.to_csv(
    "processed/diabetes_cleaned.csv",
    index=False
)

print(
    "\nDiabetes preprocessing completed!"
)

print(
    "\nSaved as:"
)

print(
    "processed/diabetes_cleaned.csv"
)

# ==========================
# FEATURE SELECTION
# ==========================

output_path = Path("processed")
output_path.mkdir(parents=True, exist_ok=True)

diabetes_features = [
    "Pregnancies",
    "Glucose",
    "BMI",
    "Age",
    "DiabetesPedigreeFunction",
]

diabetes_target = "Outcome"

selected = [c for c in diabetes_features if c in df.columns]
missing = [c for c in diabetes_features if c not in df.columns]

df_features = df.reindex(columns=selected)

for c in missing:
    df_features[c] = np.nan

if diabetes_target in df.columns:
    df_features[diabetes_target] = df[diabetes_target]
else:
    df_features[diabetes_target] = np.nan

df_features.to_csv(
    output_path / "diabetes_features.csv",
    index=False
)

print("\nSaved feature-selected file:")
print(output_path / "diabetes_features.csv")