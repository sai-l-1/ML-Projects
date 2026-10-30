import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from metrics_utils import compute_metrics

MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "support_vector_machine",
}

st.set_page_config(page_title="Breast Cancer Classifier Comparison", layout="wide")
st.title("Breast Cancer Diagnosis: Model Comparison")
st.caption("Evaluate five pre-trained classifiers on a test CSV. No retraining happens here.")


@st.cache_resource
def load_model(name):
    return joblib.load(f"model/{MODEL_FILES[name]}.joblib")


@st.cache_data
def load_meta():
    with open("model/meta.json") as f:
        return json.load(f)


meta = load_meta()
target = meta["target"]
class_names = meta["class_names"]

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV (features + target)", type="csv")
    choice = st.selectbox("Select model", list(MODEL_FILES))
    st.caption(f"Target column: **{target}** (0 = {class_names[0]}, 1 = {class_names[1]})")

# ---------------- data ----------------
if uploaded is None:
    df = pd.read_csv("test_data.csv")
    st.info("No file uploaded, using the default test_data.csv from the repository.")
else:
    df = pd.read_csv(uploaded)

if target not in df.columns:
    st.error(f"Target column '{target}' not found in the uploaded file.")
    st.stop()

try:
    y_true = df[target].astype(int).to_numpy()
except (ValueError, TypeError):
    st.error("Target column must contain 0/1 values.")
    st.stop()

if not set(np.unique(y_true)) <= set(meta["classes"]):
    st.error(f"Target values must be in {meta['classes']}.")
    st.stop()

st.subheader(f"Data: {df.shape[0]} rows x {df.shape[1]} columns")
st.dataframe(df.head())


def run_model(name):
    """Return (pipeline, predictions, probabilities) using the model's own feature names."""
    pipe = load_model(name)
    cols = list(pipe.feature_names_in_)
    absent = [c for c in cols if c not in df.columns]
    if absent:
        st.error(f"Uploaded file is missing columns the model needs: {absent}")
        st.stop()
    X = df[cols]
    return pipe, pipe.predict(X), pipe.predict_proba(X)


pipe, y_pred, y_prob = run_model(choice)
scores = compute_metrics(y_true, y_pred, y_prob)

# ---------------- metrics ----------------
st.subheader(f"Evaluation metrics: {choice}")
for col, (name, val) in zip(st.columns(len(scores)), scores.items()):
    col.metric(name, f"{val:.4f}")

# ---------------- confusion matrix + report ----------------
left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(confusion_matrix(y_true, y_pred, labels=meta["classes"]),
                annot=True, fmt="d", cmap="Purples",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    report = classification_report(y_true, y_pred, labels=meta["classes"],
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(3))

# ---------------- extra: probability distribution ----------------
st.subheader("Predicted probability of Benign")
fig2, ax2 = plt.subplots(figsize=(7, 3))
for cls, label in zip(meta["classes"], class_names):
    ax2.hist(y_prob[y_true == cls, 1], bins=15, alpha=0.6, label=f"Actual {label}")
ax2.set_xlabel("P(Benign)")
ax2.set_ylabel("Count")
ax2.legend()
st.pyplot(fig2)

# ---------------- extra: per-row predictions ----------------
pred_table = pd.DataFrame({
    "Actual": [class_names[i] for i in y_true],
    "Predicted": [class_names[i] for i in y_pred],
    "P(Benign)": y_prob[:, 1].round(4),
})
pred_table["Correct"] = pred_table["Actual"] == pred_table["Predicted"]
with st.expander("Row-level predictions"):
    st.dataframe(pred_table)
    st.download_button("Download predictions (CSV)",
                       pred_table.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODEL_FILES[choice]}.csv")

# ---------------- all models ----------------
st.subheader("All models on this data")
rows = []
for m in MODEL_FILES:
    _, p, pr = run_model(m)
    rows.append({"ML Model Name": m, **compute_metrics(y_true, p, pr)})
table = pd.DataFrame(rows).set_index("ML Model Name")
st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#d8c8f0"))
