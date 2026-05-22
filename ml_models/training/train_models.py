import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

try:
	from xgboost import XGBClassifier
except Exception:
	XGBClassifier = None


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = Path(__file__).resolve().parent / "results.csv"

DATASETS = {
	"heart": {
		"path": PROCESSED / "heart_features.csv",
		"target": "num",
	},
	"diabetes": {
		"path": PROCESSED / "diabetes_features.csv",
		"target": "Outcome",
	},
	"bp": {
		"path": PROCESSED / "bp_features.csv",
		"target": "bp_risk",
	},
}

CLASSIFIERS = {
	"logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
	"random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
	"svm": SVC(kernel="rbf", probability=True, random_state=42),
}

if XGBClassifier is not None:
	CLASSIFIERS["xgboost"] = XGBClassifier(eval_metric="logloss", random_state=42)

HEART_XGB = XGBClassifier(
	eval_metric="logloss",
	random_state=42,
	n_estimators=400,
	max_depth=3,
	learning_rate=0.05,
	subsample=0.9,
	colsample_bytree=0.9,
)

DIABETES_LR_THRESHOLD = 0.35


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, estimator, threshold=0.5):
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


def prepare_preprocessor(X: pd.DataFrame):
	numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
	categorical_cols = [c for c in X.columns if c not in numeric_cols]

	numeric_transformer = Pipeline([
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler()),
	])

	categorical_transformer = Pipeline([
		("imputer", SimpleImputer(strategy="most_frequent")),
		("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
	])

	preprocessor = ColumnTransformer([
		("num", numeric_transformer, numeric_cols),
		("cat", categorical_transformer, categorical_cols),
	], remainder="drop")

	return preprocessor


def get_roc_auc(model, X_test, y_true):
	if hasattr(model, "predict_proba"):
		scores = model.predict_proba(X_test)[:, 1]
	elif hasattr(model, "decision_function"):
		scores = model.decision_function(X_test)
	else:
		return np.nan

	return roc_auc_score(y_true, scores)


def train_dataset(name, info):
	path = info["path"]
	target = info["target"]

	if not path.exists():
		print(f"Skipping {name}: {path} not found")
		return []

	df = pd.read_csv(path)

	if target not in df.columns:
		print(f"Skipping {name}: target column '{target}' not in {path}")
		return []

	df = df.copy()
	# drop rows with missing target
	df = df[~df[target].isnull()]

	X = df.drop(columns=[target])
	y = df[target]

	# drop cols in X that are entirely NaN (e.g., requested but absent features)
	all_nan_cols = [c for c in X.columns if X[c].isnull().all()]
	if all_nan_cols:
		print(f"Dropping columns with all-NaN for {name}: {all_nan_cols}")
		X = X.drop(columns=all_nan_cols)

	# if target has only one class, skip training
	if y.nunique() < 2:
		print(f"Skipping {name}: target '{target}' has only one class: {y.unique()}")
		return []

	# convert booleans to ints
	for col in X.select_dtypes(include=["bool"]).columns:
		X[col] = X[col].astype(int)

	preprocessor = prepare_preprocessor(X)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y))>1 else None
	)

	# For heart dataset we use a feature selector after preprocessing.
	selector = None
	if name == "heart":
		selector = SelectKBest(score_func=f_classif, k=8)

	results = []

	for clf_name, clf in CLASSIFIERS.items():
		print(f"Training {clf_name} on {name}...")
		model = HEART_XGB if (name == "heart" and clf_name == "xgboost") else clf
		if name == "diabetes" and clf_name == "logistic_regression":
			model = ThresholdClassifier(
				LogisticRegression(max_iter=5000, random_state=42, class_weight="balanced"),
				threshold=DIABETES_LR_THRESHOLD,
			)

		# Build a pipeline that includes preprocessing (and selector for heart),
		# so the saved model can accept raw feature inputs from the API.
		steps = [("preprocessor", preprocessor)]
		if selector is not None:
			steps.append(("selector", selector))
		steps.append(("clf", model))

		pipeline = Pipeline(steps)

		pipeline.fit(X_train, y_train)

		preds = pipeline.predict(X_test)
		acc = accuracy_score(y_test, preds)
		precision = precision_score(y_test, preds, zero_division=0)
		recall = recall_score(y_test, preds, zero_division=0)
		f1 = f1_score(y_test, preds, zero_division=0)
		roc_auc = get_roc_auc(pipeline, X_test, y_test)

		print(
			f"{name} - {clf_name} accuracy: {acc:.4f} "
			f"precision: {precision:.4f} recall: {recall:.4f} "
			f"f1: {f1:.4f} roc_auc: {roc_auc:.4f}"
		)

		model_file = MODELS_DIR / f"{name}_{clf_name}.joblib"
		joblib.dump(pipeline, model_file)

		results.append({
			"dataset": name,
			"model": clf_name,
			"accuracy": acc,
			"precision": precision,
			"recall": recall,
			"f1_score": f1,
			"roc_auc": roc_auc,
			"model_file": str(model_file),
		})

	return results


def main():
	all_results = []

	for name, info in DATASETS.items():
		res = train_dataset(name, info)
		all_results.extend(res)

	if all_results:
		df_res = pd.DataFrame(all_results)
		df_res.to_csv(RESULTS_CSV, index=False)
		print(f"Saved results to {RESULTS_CSV}")

		comparison_dir = RESULTS_CSV.parent / "comparison_results"
		comparison_dir.mkdir(parents=True, exist_ok=True)
		for dataset, group in df_res.groupby("dataset"):
			group = group.sort_values("accuracy", ascending=False)
			group.to_csv(comparison_dir / f"{dataset}_results.csv", index=False)
			print(f"Saved {dataset} comparison results to {comparison_dir / f'{dataset}_results.csv'}")
	else:
		print("No results to save.")


if __name__ == "__main__":
	main()
