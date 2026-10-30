from pathlib import Path

import joblib
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)


# --------------------------------------------------
# 1. Create project directories
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
RESULTS_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load Breast Cancer Wisconsin dataset
# --------------------------------------------------

print("Loading Breast Cancer Wisconsin dataset...")

df = pd.read_csv("Breast Cancer Dataset.csv")
# dataset = load_breast_cancer(as_frame=True)

#df = dataset.frame.copy()
# Rename diagnosis column to target
df.rename(columns={"diagnosis": "target"}, inplace=True)

# Encode target (M=Malignant=0, B=Benign=1)
df["target"] = df["target"].map({"M": 0, "B": 1})

# Drop non-feature columns
df.drop(columns=["id"], inplace=True, errors="ignore")

# Drop unnamed index columns if present
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# The dataset contains 30 input features and one target.
print("\nOriginal dataset shape:", df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nClass distribution:")
print(df["target"].value_counts())


# --------------------------------------------------
# 3. Data cleaning
# --------------------------------------------------

print("\nStarting data cleaning...")

# Replace infinite values with NaN.
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows containing missing target labels.
df.dropna(subset=["target"], inplace=True)

# Remove duplicate rows.
duplicate_count = int(df.duplicated().sum())

print("Duplicate rows found:", duplicate_count)

df.drop_duplicates(inplace=True)

# Convert feature columns to numeric values.
feature_columns = [
    column for column in df.columns if column != "target"
]
print("Number of features:", len(feature_columns))

for column in feature_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

# Remove rows containing invalid or missing target values.
df = df[df["target"].isin([0, 1])].copy()

# Convert target to integer.
df["target"] = df["target"].astype(int)

# Check feature values for missing data.
print("\nMissing values before model imputation:")
print(df[feature_columns].isna().sum().sum())

# Check for any remaining invalid values.
print("\nDataset shape after cleaning:", df.shape)

print("\nClass distribution after cleaning:")
print(df["target"].value_counts())

# Verify the cleaned dataset is not empty.
if df.empty:
    raise ValueError("The dataset is empty after cleaning.")

if df["target"].nunique() != 2:
    raise ValueError(
        "The dataset must contain both target classes."
    )

# Confirm that the expected feature columns are present.
if len(feature_columns) != 30:
    raise ValueError(
        f"Expected 30 features, found {len(feature_columns)}."
    )


# --------------------------------------------------
# 4. Separate input features and target
# --------------------------------------------------

X = df[feature_columns].copy()
y = df["target"].copy()

# Scikit-learn's target encoding:
# 0 = malignant
# 1 = benign

print("\nTarget mapping:")
print("0 = malignant")
print("1 = benign")


# --------------------------------------------------
# 5. Split the dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 6. Save held-out test data
# --------------------------------------------------

test_data = X_test.copy()

# Include the actual target so evaluation metrics
# can be calculated by the Streamlit application.
test_data["target"] = y_test

test_data.to_csv(
    BASE_DIR / "test_data.csv",
    index=False,
)

print("\nTest data saved to test_data.csv")


# --------------------------------------------------
# 7. Define preprocessing and classification models
# --------------------------------------------------

def make_scaled_pipeline(classifier):
    """
    Create a pipeline that imputes missing values,
    scales features, and applies a classifier.

    Preprocessing is fitted only on training data.
    """

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            classifier,
        ),
    ])


models = {
    "Logistic Regression": make_scaled_pipeline(
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        )
    ),

    "Decision Tree": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                random_state=42,
            ),
        ),
    ]),

    "KNN": make_scaled_pipeline(
        KNeighborsClassifier(
            n_neighbors=5,
        )
    ),

    "Naive Bayes": Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "classifier",
            GaussianNB(),
        ),
    ]),

    "Support Vector Machine": make_scaled_pipeline(
        SVC(
            probability=True,
            random_state=42,
        )
    ),
}


# --------------------------------------------------
# 8. Train models and calculate evaluation metrics
# --------------------------------------------------

results = []

print("\nTraining models...")

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    # Fit using training data only.
    model.fit(X_train, y_train)

    # Predict labels for the held-out test data.
    y_pred = model.predict(X_test)

    # Obtain probability estimates for AUC.
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "ML Model Name": model_name,

        "Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),

        "AUC": roc_auc_score(
            y_test,
            y_prob,
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),

        "F1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),

        "MCC": matthews_corrcoef(
            y_test,
            y_pred,
        ),
    }

    results.append(metrics)

    # Save the complete fitted pipeline.
    filename = (
        model_name.lower()
        .replace(" ", "_")
        + ".joblib"
    )

    model_path = MODEL_DIR / filename

    joblib.dump(model, model_path)

    print("Saved model:", model_path)

    for metric_name, metric_value in metrics.items():

        if metric_name != "ML Model Name":
            print(
                f"{metric_name}: {metric_value:.4f}"
            )


# --------------------------------------------------
# 9. Save model comparison results
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = RESULTS_DIR / "model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False,
)

print("\nFinal model comparison:")
print(results_df.round(4).to_string(index=False))

print("\nResults saved to:", results_path)


# --------------------------------------------------
# 10. Display cleaning summary
# --------------------------------------------------

print("\nData cleaning and model training completed.")

print("Final cleaned dataset shape:", df.shape)

print(
    "Remaining missing feature values:",
    int(df[feature_columns].isna().sum().sum()),
)

print("Saved models:", len(models))
print("Evaluation metrics per model: 6")
print("Total evaluation results:", len(models) * 6)
