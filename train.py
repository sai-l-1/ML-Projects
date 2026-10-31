"""Train five classifiers to predict a student's performance level from lifestyle
habits."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from metrics_utils import compute_metrics

# ------------------------------------------------------------------
# 1. Paths and constants (relative to this file)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"
DATA_FILE = BASE_DIR / "student_habits_performance.csv"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
TARGET = "performance_level"
LEVELS = ["Low", "Medium", "High"]          # encoded 0, 1, 2

# ------------------------------------------------------------------
# 2. Load data and build the classification target
#    exam_score (0-100) -> Low (<50), Medium (50-<70), High (>=70)
# ------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
print("Raw shape:", df.shape)

df = df.drop(columns=["student_id"])                      # identifier, not a feature
df["score_band"] = pd.cut(df["exam_score"], bins=[-np.inf, 50, 70, np.inf],
                          labels=[0, 1, 2], right=False).astype(int)
df = df.drop(columns=["exam_score"])                      # avoid target leakage
df = df.rename(columns={"score_band": TARGET})

df = df.drop_duplicates()
feature_cols = [c for c in df.columns if c != TARGET]
num_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

assert len(feature_cols) >= 12 and len(df) >= 500
print("Rows:", len(df), "| Features:", len(feature_cols))
print("Numeric:", num_cols)
print("Categorical:", cat_cols)
print("Missing values per column:\n", df.isna().sum()[df.isna().sum() > 0])
print("Class counts:\n", df[TARGET].map(dict(enumerate(LEVELS))).value_counts())

# ------------------------------------------------------------------
# 3. Split and save the held-out test set (human-readable labels)
# ------------------------------------------------------------------
X, y = df[feature_cols], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)

test_data = X_test.copy()
test_data[TARGET] = y_test.map(dict(enumerate(LEVELS)))
test_data.to_csv(BASE_DIR / "test_data.csv", index=False)
print("Train:", len(X_train), "| Test:", len(X_test))

# ------------------------------------------------------------------
# 4. Metadata used by the app (same folder as the models)
# ------------------------------------------------------------------
meta = {
    "target": TARGET,
    "levels": LEVELS,
    "features": feature_cols,
    "numeric": {c: {"min": float(df[c].min()), "max": float(df[c].max()),
                    "median": float(df[c].median()),
                    "is_int": bool((df[c].dropna() % 1 == 0).all())}
                for c in num_cols},
    "categorical": {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols},
}
with open(MODEL_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=4)

# ------------------------------------------------------------------
# 5. Preprocessing + models
# ------------------------------------------------------------------
def preprocessor(scale_numeric):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]), cat_cols),
    ])

def build(clf, scale_numeric=True):
    return Pipeline([("prep", preprocessor(scale_numeric)), ("model", clf)])

models = {
    "Logistic Regression": build(LogisticRegression(max_iter=3000, random_state=SEED)),
    "Decision Tree": build(DecisionTreeClassifier(max_depth=5, min_samples_leaf=5,
                                                  random_state=SEED), scale_numeric=False),
    "KNN": build(KNeighborsClassifier(n_neighbors=9)),
    # var_smoothing raised from the 1e-9 default: one-hot columns can have ~zero variance
    # within a class, as it can make GaussianNB overconfident
    "Naive Bayes": build(GaussianNB(var_smoothing=1e-3), scale_numeric=False),
    "Support Vector Machine": build(SVC(kernel="rbf", C=1.0, probability=True,
                                        random_state=SEED)),
}

# ------------------------------------------------------------------
# 6. Train, evaluate, save
# ------------------------------------------------------------------
labels = [0, 1, 2]
rows = []
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    scores = compute_metrics(y_test, pipe.predict(X_test),
                             pipe.predict_proba(X_test), labels)
    rows.append({"ML Model Name": name, **scores})
    joblib.dump(pipe, MODEL_DIR / (name.lower().replace(" ", "_") + ".joblib"))
    print(f"Saved {name}")

results = pd.DataFrame(rows)
results.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
print("\n", results.round(4).to_string(index=False))
