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

st.set_page_config(page_title="Student Performance Predictor", layout="wide")
st.title("Student Performance Level: Model Comparison")
st.caption("Predicts Low / Medium / High exam performance from daily habits. "
           "Models are pre-trained; nothing is retrained here.")


@st.cache_resource
def load_model(name):
    return joblib.load(f"model/{MODEL_FILES[name]}.joblib")


@st.cache_data
def load_meta():
    with open("model/meta.json") as f:
        return json.load(f)


meta = load_meta()
target, levels = meta["target"], meta["levels"]
level_to_id = {lvl: i for i, lvl in enumerate(levels)}
labels = list(range(len(levels)))

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload test CSV (features + target)", type="csv")
    choice = st.selectbox("Select model", list(MODEL_FILES))
    st.caption(f"Target column: **{target}** with values {', '.join(levels)}")

tab_eval, tab_single = st.tabs(["Evaluate on test data", "Try a single student"])

# ======================= TAB 1: evaluation =======================
with tab_eval:
    if uploaded is None:
        df = pd.read_csv("test_data.csv")
        st.info("No file uploaded, using the default test_data.csv from the repository.")
    else:
        df = pd.read_csv(uploaded)

    if target not in df.columns:
        st.error(f"Target column '{target}' not found.")
        st.stop()
    bad = set(df[target].astype(str)) - set(levels)
    if bad:
        st.error(f"Unknown target labels: {bad}. Expected {levels}.")
        st.stop()
    y_true = df[target].astype(str).map(level_to_id).to_numpy()

    st.subheader(f"Data: {df.shape[0]} rows x {df.shape[1]} columns")
    st.dataframe(df.head())

    def run_model(name):
        pipe = load_model(name)
        cols = list(pipe.feature_names_in_)
        absent = [c for c in cols if c not in df.columns]
        if absent:
            st.error(f"Uploaded file is missing columns: {absent}")
            st.stop()
        X = df[cols]
        return pipe.predict(X), pipe.predict_proba(X)

    y_pred, y_prob = run_model(choice)
    scores = compute_metrics(y_true, y_pred, y_prob, labels)

    st.subheader(f"Evaluation metrics: {choice}")
    st.caption("Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro.")
    for col, (k, v) in zip(st.columns(len(scores)), scores.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(confusion_matrix(y_true, y_pred, labels=labels), annot=True,
                    fmt="d", cmap="YlGnBu", xticklabels=levels,
                    yticklabels=levels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        rep = classification_report(y_true, y_pred, labels=labels, target_names=levels,
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3))

    st.subheader("Average predicted probability per actual level")
    avg = pd.DataFrame(y_prob, columns=levels).groupby(
        pd.Series(y_true).map(dict(enumerate(levels))).values).mean().reindex(levels)
    st.bar_chart(avg)

    st.subheader("All models on this data")
    table = pd.DataFrame([
        {"ML Model Name": m, **compute_metrics(y_true, *run_model(m), labels)}
        for m in MODEL_FILES]).set_index("ML Model Name")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#cfe8e0"))

# ======================= TAB 2: single student =======================
with tab_single:
    st.write(f"Enter a student's habits and see what **{choice}** predicts.")
    pipe = load_model(choice)
    values = {}
    cols = st.columns(3)
    for i, feat in enumerate(pipe.feature_names_in_):
        box = cols[i % 3]
        if feat in meta["numeric"]:
            info = meta["numeric"][feat]
            if info["is_int"]:
                values[feat] = box.number_input(feat, int(info["min"]), int(info["max"]),
                                                int(round(info["median"])), step=1)
            else:
                values[feat] = box.number_input(feat, float(info["min"]),
                                                float(info["max"]), float(info["median"]))
        else:
            values[feat] = box.selectbox(feat, meta["categorical"][feat])
    if st.button("Predict performance level"):
        row = pd.DataFrame([values])[list(pipe.feature_names_in_)]
        proba = pipe.predict_proba(row)[0]
        st.success(f"Predicted level: **{levels[int(np.argmax(proba))]}**")
        st.bar_chart(pd.Series(proba, index=levels, name="Probability"))
