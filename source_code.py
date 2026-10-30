"""Training script: cleans data, trains 5 classifiers, saves models, test split
and metadata"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from metrics_utils import compute_metrics

# ---------------------------------------------------------------
# 1. Paths (all relative to this file, so it works on any machine)
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"
DATA_FILE = BASE_DIR / "Breast Cancer Dataset.csv"
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

TARGET = "target"
CLASS_NAMES = {0: "Malignant", 1: "Benign"}
SEED = 42

# ---------------------------------------------------------------
# 2. Load and prepare the dataset
# ---------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_FILE)
df = df.rename(columns={"diagnosis": TARGET})
df[TARGET] = df[TARGET].map({"M": 0, "B": 1})
df = df.drop(columns=["id"], errors="ignore")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("Shape:", df.shape)
print(df[TARGET].value_counts())

# ---------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[TARGET])
print("Duplicate rows:", int(df.duplicated().sum()))
df = df.drop_duplicates()

feature_cols = [c for c in df.columns if c != TARGET]
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[TARGET] = df[TARGET].astype(int)

assert len(feature_cols) >= 12, "Need at least 12 features"
assert len(df) >= 500, "Need at least 500 instances"
assert df[TARGET].nunique() == 2, "Both classes must be present"
print("Features:", len(feature_cols), "| Rows after cleaning:", len(df))

# ---------------------------------------------------------------
# 4. Split + save held-out test data
# ---------------------------------------------------------------
X, y = df[feature_cols], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

test_data = X_test.copy()
test_data[TARGET] = y_test
test_data.to_csv(BASE_DIR / "test_data.csv", index=False)
print("Train:", len(X_train), "| Test:", len(X_test))

# ---------------------------------------------------------------
# 5. Metadata for the app (relative path, same folder as models)
# ---------------------------------------------------------------
meta = {
    "target": TARGET,
    "features": feature_cols,
    "classes": sorted(CLASS_NAMES),
    "class_names": [CLASS_NAMES[k] for k in sorted(CLASS_NAMES)],
}
with open(MODEL_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=4)

# ---------------------------------------------------------------
# 6. Models
# ---------------------------------------------------------------
def scaled(clf):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("classifier", clf)])

def unscaled(clf):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("classifier", clf)])

models = {
    "Logistic Regression": scaled(LogisticRegression(max_iter=2000, random_state=SEED)),
    "Decision Tree": unscaled(DecisionTreeClassifier(random_state=SEED)),
    "KNN": scaled(KNeighborsClassifier(n_neighbors=5)),
    "Naive Bayes": unscaled(GaussianNB()),
    "Support Vector Machine": scaled(SVC(probability=True, random_state=SEED)),
}

# ---------------------------------------------------------------
# 7. Train, evaluate, save
# ---------------------------------------------------------------
rows = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    scores = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test))
    rows.append({"ML Model Name": name, **scores})

    fname = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, MODEL_DIR / fname)
    print("Saved:", MODEL_DIR / fname)
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")

results_df = pd.DataFrame(rows)
results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
print("\nFinal comparison:")
print(results_df.round(4).to_string(index=False))
print("\nDone. Models saved:", len(models))
