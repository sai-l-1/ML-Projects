import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from metrics_utils import compute_metrics

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "support_vector_machine",
}
MODEL_BLURBS = {
    "Logistic Regression": "Linear model that weighs each habit and turns the total into class probabilities.",
    "Decision Tree": "Rule-based model that splits students by thresholds (for example study hours).",
    "KNN": "Compares a student with the 9 most similar students in the training data.",
    "Naive Bayes": "Probabilistic model that combines each habit's evidence independently.",
    "Support Vector Machine": "Finds a smooth boundary between the levels using an RBF kernel.",
}
MODEL_ICONS = {"Logistic Regression": "📈", "Decision Tree": "🌳", "KNN": "👥",
               "Naive Bayes": "🎲", "Support Vector Machine": "🧭"}
LEVEL_COLORS = {"Low": "#ef476f", "Medium": "#ffb703", "High": "#06a77d"}

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# ------------------------------------------------------------------
# Custom styling
# ------------------------------------------------------------------
st.markdown("""
<style>
.hero {background: linear-gradient(120deg, #3a0ca3 0%, #4361ee 55%, #4cc9f0 100%);
       padding: 2rem 2.2rem; border-radius: 18px; color: #fff; margin-bottom: 1.2rem;}
.hero h1 {margin: 0 0 .3rem 0; font-size: 2.1rem; color: #fff;}
.hero p {margin: 0; font-size: 1.02rem; opacity: .95;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 1rem 1.1rem;
       height: 100%; background: rgba(67,97,238,.06);}
.card h4 {margin: 0 0 .35rem 0;}
.card p {margin: 0; font-size: .9rem; opacity: .85;}
.step {border-left: 5px solid #4361ee; padding: .5rem .9rem; margin-bottom: .6rem;
       background: rgba(67,97,238,.06); border-radius: 0 10px 10px 0;}
.pill {display: inline-block; padding: .15rem .65rem; border-radius: 999px; color: #fff;
       font-weight: 600; font-size: .85rem; margin-right: .3rem;}
[data-testid="stMetric"] {background: rgba(67,97,238,.08); border: 1px solid rgba(67,97,238,.25);
       padding: .7rem .9rem; border-radius: 12px;}
.section-title {font-size: 1.35rem; font-weight: 700; margin: 1.4rem 0 .6rem 0;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------
@st.cache_resource
def load_model(name):
    return joblib.load(f"model/{MODEL_FILES[name]}.joblib")


@st.cache_data
def load_meta():
    with open("model/meta.json") as f:
        return json.load(f)


meta = load_meta()
target, levels, features = meta["target"], meta["levels"], meta["features"]
level_to_id = {lvl: i for i, lvl in enumerate(levels)}
labels = list(range(len(levels)))

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Controls")
    uploaded = st.file_uploader("1. Upload test CSV (features + target)", type="csv")
    choice = st.selectbox("2. Select model to inspect", list(MODEL_FILES))
    st.divider()
    st.caption(f"Target column: **{target}**  \nValues: {', '.join(levels)}")
    sample_path = Path("test_data.csv")
    if sample_path.exists():
        st.download_button("⬇ Download sample test CSV", sample_path.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

# ------------------------------------------------------------------
# Hero
# ------------------------------------------------------------------
st.markdown(f"""
<div class="hero">
  <h1>🎓 Student Performance Level Predictor</h1>
  <p>Compare five pre-trained machine-learning models that predict whether a student's exam
  performance is <b>Low</b>, <b>Medium</b> or <b>High</b> from daily habits.
  Models are loaded from disk; nothing is retrained here.</p>
</div>
""", unsafe_allow_html=True)

tab_eval, tab_single = st.tabs(["📊 Evaluate on test data", "🧑‍🎓 Try a single student"])


# ------------------------------------------------------------------
# Landing page (shown until a CSV is uploaded)
# ------------------------------------------------------------------
def show_landing():
    st.info("👈 Upload a test CSV in the sidebar to see the results. "
            "Everything below explains what the app uses and what it expects.")

    st.markdown('<div class="section-title">🤖 Models used</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for col, (name, blurb) in zip(cols, MODEL_BLURBS.items()):
        col.markdown(f'<div class="card"><h4>{MODEL_ICONS[name]} {name}</h4><p>{blurb}</p></div>',
                     unsafe_allow_html=True)

    st.markdown('<div class="section-title">🎯 Prediction classes</div>', unsafe_allow_html=True)
    rules = {"Low": "exam score below 50", "Medium": "exam score from 50 to under 70",
             "High": "exam score 70 or above"}
    st.markdown("".join(
        f'<span class="pill" style="background:{LEVEL_COLORS[l]}">{l}</span> {rules[l]} &nbsp;&nbsp;'
        for l in levels), unsafe_allow_html=True)

    st.markdown('<div class="section-title">📦 What you will see after uploading</div>',
                unsafe_allow_html=True)
    outputs = [
        ("Evaluation metrics", "Accuracy, AUC, Precision, Recall, F1 and MCC for the model chosen in the sidebar."),
        ("Confusion matrix", "How many students of each actual level were predicted as each level."),
        ("Classification report", "Precision, recall, F1 and support for Low, Medium and High."),
        ("Average probabilities", "Average confidence the model gives to each level, per actual level."),
        ("All-models comparison", "The six metrics for all five models side by side, with the best value per column highlighted."),
        ("Row-level predictions", "Prediction and confidence for every uploaded student, with CSV download."),
    ]
    c1, c2 = st.columns(2)
    for i, (t, d) in enumerate(outputs):
        (c1 if i % 2 == 0 else c2).markdown(f'<div class="step"><b>{t}</b><br>{d}</div>',
                                            unsafe_allow_html=True)

    st.markdown('<div class="section-title">🗂 Required test-data fields</div>', unsafe_allow_html=True)
    st.write(f"The CSV needs **{len(features)} feature columns** plus the target column "
             f"`{target}`. Column names must match exactly; extra columns are ignored.")
    rows = []
    for f in features:
        if f in meta["numeric"]:
            n = meta["numeric"][f]
            kind = "Whole number" if n["is_int"] else "Decimal number"
            fmt = (lambda v: f"{int(v)}") if n["is_int"] else (lambda v: f"{v:g}")
            allowed = f"{fmt(n['min'])} to {fmt(n['max'])}"
        else:
            kind = "Text"
            allowed = ", ".join(map(str, meta["categorical"][f]))
        rows.append({"Column": f, "Type": kind, "Allowed values / range": allowed})
    rows.append({"Column": target, "Type": "Text (target)",
                 "Allowed values / range": " / ".join(levels)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Missing values are allowed in feature columns (they are imputed by the models). "
               "The target column must have no blanks and only the three values above.")

    template = pd.DataFrame(columns=features + [target])
    st.download_button("⬇ Download empty CSV template (headers only)",
                       template.to_csv(index=False).encode(), file_name="test_data_template.csv",
                       mime="text/csv")


# ------------------------------------------------------------------
# Results (shown only after upload)
# ------------------------------------------------------------------
def show_results(df):
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        st.error(f"Your file is missing required columns: {missing}")
        return
    bad = set(df[target].dropna().astype(str)) - set(levels)
    if bad or df[target].isna().any():
        st.error(f"Target column `{target}` must contain only {levels}. Found invalid values: "
                 f"{sorted(bad) if bad else 'blank cells'}.")
        return

    y_true = df[target].astype(str).map(level_to_id).to_numpy()
    st.success(f"Loaded {df.shape[0]} students with {len(features)} features. Showing results for "
               f"**{choice}** (change it in the sidebar).")
    with st.expander("Preview uploaded data"):
        st.dataframe(df.head(20), use_container_width=True)

    def run_model(name):
        pipe = load_model(name)
        X = df[list(pipe.feature_names_in_)]
        return pipe.predict(X), pipe.predict_proba(X)

    y_pred, y_prob = run_model(choice)
    scores = compute_metrics(y_true, y_pred, y_prob, labels)

    st.markdown(f'<div class="section-title">📏 Evaluation metrics: {choice}</div>',
                unsafe_allow_html=True)
    st.caption("Precision, Recall and F1 are macro-averaged; AUC is one-vs-rest macro.")
    for col, (k, v) in zip(st.columns(len(scores)), scores.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="section-title">🧮 Confusion matrix</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(4.8, 3.9))
        sns.heatmap(confusion_matrix(y_true, y_pred, labels=labels), annot=True, fmt="d",
                    cmap="Blues", xticklabels=levels, yticklabels=levels, cbar=False, ax=ax)
        ax.set_xlabel("Predicted level")
        ax.set_ylabel("Actual level")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown('<div class="section-title">📝 Classification report</div>', unsafe_allow_html=True)
        rep = classification_report(y_true, y_pred, labels=labels, target_names=levels,
                                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), use_container_width=True)

    st.markdown('<div class="section-title">📊 Average predicted probability per actual level</div>',
                unsafe_allow_html=True)
    avg = pd.DataFrame(y_prob, columns=levels)
    avg["Actual"] = [levels[i] for i in y_true]
    st.bar_chart(avg.groupby("Actual").mean().reindex(levels))

    st.markdown('<div class="section-title">🏆 All models on this data</div>', unsafe_allow_html=True)
    table = pd.DataFrame([{"ML Model Name": m, **compute_metrics(y_true, *run_model(m), labels)}
                          for m in MODEL_FILES]).set_index("ML Model Name")
    best = table["MCC"].idxmax()
    st.success(f"Best model on this data by MCC: **{best}** "
               f"(MCC {table.loc[best, 'MCC']:.4f}, accuracy {table.loc[best, 'Accuracy']:.4f})")
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#b7e4c7"),
                 use_container_width=True)
    st.download_button("⬇ Download comparison table (CSV)", table.to_csv().encode(),
                       file_name="model_comparison_uploaded.csv", mime="text/csv")

    with st.expander(f"Row-level predictions ({choice})"):
        pred = pd.DataFrame({
            "Actual": [levels[i] for i in y_true],
            "Predicted": [levels[i] for i in y_pred],
            "Confidence": y_prob.max(axis=1).round(4),
        })
        pred["Correct"] = pred["Actual"] == pred["Predicted"]
        st.dataframe(pred, use_container_width=True)
        st.download_button("⬇ Download predictions (CSV)", pred.to_csv(index=False).encode(),
                           file_name=f"predictions_{MODEL_FILES[choice]}.csv", mime="text/csv")


with tab_eval:
    if uploaded is None:
        show_landing()
    else:
        try:
            data = pd.read_csv(uploaded)
        except Exception as e:
            data = None
            st.error(f"Could not read the uploaded file as CSV: {e}")
        if data is not None:
            show_results(data)

# ------------------------------------------------------------------
# Single student tab
# ------------------------------------------------------------------
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
    if st.button("Predict performance level", type="primary"):
        row = pd.DataFrame([values])[list(pipe.feature_names_in_)]
        proba = pipe.predict_proba(row)[0]
        top = levels[int(np.argmax(proba))]
        st.markdown(f'Predicted level: <span class="pill" style="background:{LEVEL_COLORS[top]}">'
                    f'{top}</span> (confidence {proba.max():.1%})', unsafe_allow_html=True)
        st.bar_chart(pd.Series(proba, index=levels, name="Probability"))
