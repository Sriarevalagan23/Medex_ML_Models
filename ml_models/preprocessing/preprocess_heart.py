import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv(
    "datasets/heart.csv"
)

print("\nOriginal Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values Before:")
print(df.isnull().sum())


# ==========================
# DROP UNUSED COLUMNS
# ==========================

columns_to_drop = [
    "id",
    "origin"
]

existing_columns = [
    col for col
    in columns_to_drop
    if col in df.columns
]

df.drop(
    columns=existing_columns,
    inplace=True
)


# ==========================
# REPLACE '?' VALUES
# ==========================

df.replace(
    "?",
    np.nan,
    inplace=True
)


# ==========================
# CONVERT NUMERIC COLUMNS
# ==========================

numeric_columns = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "ca"
]

for col in numeric_columns:

    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ==========================
# HANDLE TARGET COLUMN
# num:
# 0 = Low Risk
# 1,2,3,4 = High Risk
# ==========================

df["num"] = (
    df["num"] > 0
).astype(int)


# ==========================
# HANDLE MISSING VALUES
# ==========================

for col in df.columns:

    if (
        df[col].dtype
        in [
            "int64",
            "float64"
        ]
    ):
        df[col].fillna(
            df[col].median(),
            inplace=True
        )

    else:
        df[col].fillna(
            df[col].mode()[0],
            inplace=True
        )


# ==========================
# LABEL ENCODING
# ==========================

encoder = LabelEncoder()

categorical_columns = [
    "sex",
    "cp",
    "restecg",
    "exang",
    "slope",
    "thal"
]

for col in categorical_columns:

    if col in df.columns:

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )


# ==========================
# REMOVE DUPLICATES
# ==========================

df.drop_duplicates(
    inplace=True
)


# ==========================
# REMOVE OUTLIERS (IQR)
# ==========================

numeric_cols = df.select_dtypes(
    include=np.number
).columns

for col in numeric_cols:

    if col != "num":

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
    df["num"]
    .value_counts()
)

print(
    "\nData Types:"
)
print(
    df.dtypes
)


# ==========================
# SAVE CLEAN DATA
# ==========================

output_path = Path("processed")
output_path.mkdir(parents=True, exist_ok=True)

df.to_csv(
    output_path / "heart_cleaned.csv",
    index=False
)

print(
    "\nHeart dataset preprocessing completed!"
)

print(
    "\nSaved as:"
)

print(
    output_path / "heart_cleaned.csv"
)

# ==========================
# FEATURE SELECTION
# ==========================

heart_features = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

heart_target = "num"

selected = [c for c in heart_features if c in df.columns]
missing = [c for c in heart_features if c not in df.columns]

df_features = df.reindex(columns=selected)

for c in missing:
    df_features[c] = np.nan

if heart_target in df.columns:
    df_features[heart_target] = df[heart_target]
else:
    df_features[heart_target] = np.nan

df_features.to_csv(
    output_path / "heart_features.csv",
    index=False
)

print("\nSaved feature-selected file:")
print(output_path / "heart_features.csv")