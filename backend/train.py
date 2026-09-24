import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression

import joblib

# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = "heart.csv"
target = "HeartDisease"

df = pd.read_csv(file_path)

print("Dataset shape:", df.shape)
print("\nDataset info:")
print(df.info())


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(target, axis=1)
y = df[target]


# ============================================================
# 3. REPLACE INVALID VALUES WITH NaN
# ============================================================

X["Cholesterol"] = X["Cholesterol"].replace(0, np.nan)
X["RestingBP"] = X["RestingBP"].replace(0, np.nan)


# ============================================================
# 4. IDENTIFY NUMERIC AND CATEGORICAL COLUMNS
# ============================================================

numeric_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumeric columns:")
print(numeric_cols)

print("\nCategorical columns:")
print(categorical_cols)


# ============================================================
# 5. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. NUMERIC PIPELINE
# ============================================================

num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])


# ============================================================
# 7. CATEGORICAL PIPELINE
# ============================================================

cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


# ============================================================
# 8. COMBINE BOTH PIPELINES
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols)
    ]
)


# ============================================================
# 9. FINAL MODEL PIPELINE
# ============================================================

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])


# ============================================================
# 10. TRAIN
# ============================================================

print("\n" + "=" * 80)
print("Training model...")

model_pipeline.fit(X_train, y_train)


# ============================================================
# 11. PREDICTION
# ============================================================

pred = model_pipeline.predict(X_test)


# ============================================================
# 12. EVALUATION
# ============================================================

print("\n" + "=" * 80)

print(f"Accuracy  : {accuracy_score(y_test, pred):.4f}")
print(f"Precision : {precision_score(y_test, pred):.4f}")
print(f"Recall    : {recall_score(y_test, pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

joblib.dump(model_pipeline,"model.pkl")
print("model save in model.pkl file ")

