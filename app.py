"""CampusOutcome Lab — Streamlit front-end for student academic-outcome classifiers."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
BUNDLED_TEST_CSV = ROOT / "test_data.csv"

MODEL_CHOICES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}

CLASS_PALETTE = {
    "Dropout": "#B85C38",
    "Enrolled": "#4A7C9B",
    "Graduate": "#1F6F5B",
}

PAGE_CSS = """
<style>
    .block-container { padding-top: 1.4rem; max-width: 1180px; }
    .hero-kicker {
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.78rem;
        color: #1F6F5B;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .hero-title {
        font-size: 2.05rem;
        font-weight: 750;
        color: #1C2A24;
        line-height: 1.15;
        margin-bottom: 0.35rem;
    }
    .hero-lede {
        color: #4E5B55;
        font-size: 1.02rem;
        margin-bottom: 1.1rem;
    }
    .metric-card {
        background: #fffdf8;
        border: 1px solid #d9d0be;
        border-radius: 14px;
        padding: 0.85rem 0.9rem 0.75rem 0.9rem;
        box-shadow: 0 1px 0 rgba(28, 42, 36, 0.04);
    }
    .metric-label {
        font-size: 0.78rem;
        color: #6a746e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #1F6F5B;
        margin-top: 0.15rem;
    }
    div[data-testid="stSidebar"] {
        background: #1C2A24;
    }
    div[data-testid="stSidebar"] * {
        color: #F7F3EA !important;
    }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stRadio label,
    div[data-testid="stSidebar"] .stFileUploader label {
        color: #F7F3EA !important;
    }
</style>
"""


@st.cache_resource
def load_campus_artifacts(model_files: tuple[str, ...]):
    preprocessing = joblib.load(MODEL_DIR / "preprocessing.joblib")
    fitted = {
        name: joblib.load(MODEL_DIR / filename)
        for name, filename in MODEL_CHOICES.items()
        if filename in model_files
    }
    return preprocessing, fitted


def read_student_csv(source) -> pd.DataFrame:
    def _load(separator: str) -> pd.DataFrame:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, sep=separator)

    frame = _load(",")
    if frame.shape[1] == 1:
        frame = _load(";")
    frame.columns = [str(col).replace("\t", "").strip() for col in frame.columns]
    return frame


def score_holdout(estimator, features, true_codes) -> dict:
    predicted = estimator.predict(features)
    probabilities = estimator.predict_proba(features)
    return {
        "Accuracy": accuracy_score(true_codes, predicted),
        "AUC": roc_auc_score(
            true_codes, probabilities, multi_class="ovr", average="macro"
        ),
        "Precision": precision_score(
            true_codes, predicted, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            true_codes, predicted, average="weighted", zero_division=0
        ),
        "F1": f1_score(true_codes, predicted, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(true_codes, predicted),
    }


def render_metric_strip(scores: dict) -> None:
    labels = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    columns = st.columns(len(labels))
    for column, label in zip(columns, labels):
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{scores[label]:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def confusion_figure(true_labels, predicted_labels, class_names):
    matrix = confusion_matrix(true_labels, predicted_labels, labels=class_names)
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=class_names,
            y=class_names,
            colorscale=[
                [0.0, "#F7F3EA"],
                [0.5, "#8FB9A8"],
                [1.0, "#1F6F5B"],
            ],
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="True %{y}<br>Predicted %{x}<br>Count %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Predicted outcome",
        yaxis_title="True outcome",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=20, t=20, b=40),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="CampusOutcome Lab",
        page_icon="🎓",
        layout="wide",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    preprocessing, fitted_models = load_campus_artifacts(tuple(MODEL_CHOICES.values()))
    feature_cols = preprocessing["feature_columns"]
    class_names = list(preprocessing["class_names"])
    label_codec = preprocessing["label_encoder"]

    st.markdown('<div class="hero-kicker">BITS WILP · Machine Learning Assignment 2</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">CampusOutcome Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-lede">Predict whether a higher-education student will drop out, '
        "stay enrolled, or graduate — then inspect hold-out metrics, a confusion matrix, "
        "and a classification report for each model.</div>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Evaluation controls")
        uploaded = st.file_uploader(
            "Upload test CSV (same columns as training data)",
            type=["csv"],
            help="Streamlit Cloud has a small memory budget, so upload only a test split.",
        )
        model_name = st.selectbox("Classifier", list(MODEL_CHOICES.keys()))
        show_all = st.checkbox("Also compare every model on this file", value=True)
        st.markdown("---")
        st.caption(
            "Dataset: UCI *Predict Students' Dropout and Academic Success*  \n"
            "4,424 students · 36 features · 3 outcome classes"
        )

    if uploaded is not None:
        student_frame = read_student_csv(uploaded)
        data_caption = f"Using uploaded file: **{uploaded.name}**"
    else:
        student_frame = read_student_csv(BUNDLED_TEST_CSV)
        data_caption = "No file uploaded — scoring the bundled `test_data.csv` hold-out split."

    st.info(data_caption)

    missing = [col for col in feature_cols if col not in student_frame.columns]
    if missing:
        st.error(
            "This CSV is missing required feature columns: "
            + ", ".join(missing[:8])
            + ("…" if len(missing) > 8 else "")
        )
        st.stop()

    predictors = student_frame[feature_cols]
    has_target = "Target" in student_frame.columns

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Class mix in the loaded file")
        if has_target:
            mix = student_frame["Target"].value_counts().rename_axis("Outcome").reset_index(name="Students")
            pie = px.pie(
                mix,
                names="Outcome",
                values="Students",
                color="Outcome",
                color_discrete_map=CLASS_PALETTE,
                hole=0.45,
            )
            pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
            st.plotly_chart(pie, width="stretch")
        else:
            st.warning("No `Target` column found. Metrics cannot be computed; predictions only.")
    with right:
        st.subheader("File snapshot")
        st.write(
            {
                "rows": int(len(student_frame)),
                "features used": int(len(feature_cols)),
                "selected model": model_name,
                "has labels": bool(has_target),
            }
        )
        st.dataframe(student_frame.head(6), width="stretch", hide_index=True)

    chosen = fitted_models[model_name]
    predicted_codes = chosen.predict(predictors)
    predicted_labels = label_codec.inverse_transform(predicted_codes)

    if has_target:
        true_labels = student_frame["Target"].astype(str)
        unknown = sorted(set(true_labels) - set(class_names))
        if unknown:
            st.error(f"Unexpected Target values: {unknown}. Expected {class_names}.")
            st.stop()
        true_codes = label_codec.transform(true_labels)
        scores = score_holdout(chosen, predictors, true_codes)

        st.subheader(f"Hold-out metrics · {model_name}")
        render_metric_strip(scores)

        matrix_col, report_col = st.columns(2)
        with matrix_col:
            st.markdown("**Confusion matrix**")
            st.plotly_chart(
                confusion_figure(true_labels, predicted_labels, class_names),
                width="stretch",
            )
        with report_col:
            st.markdown("**Classification report**")
            report_text = classification_report(
                true_labels,
                predicted_labels,
                labels=class_names,
                digits=3,
                zero_division=0,
            )
            st.code(report_text, language="text")

        if show_all:
            st.subheader("All six models on this test file")
            board_rows = []
            for name, estimator in fitted_models.items():
                row = score_holdout(estimator, predictors, true_codes)
                row["ML Model Name"] = name
                board_rows.append(row)
            board = pd.DataFrame(board_rows)[
                ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
            ]
            display_board = board.copy()
            for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
                display_board[metric] = display_board[metric].map(lambda value: f"{value:.4f}")
            st.dataframe(display_board, width="stretch", hide_index=True)

            long_board = board.melt(
                id_vars="ML Model Name",
                value_vars=["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"],
                var_name="Metric",
                value_name="Score",
            )
            bars = px.bar(
                long_board,
                x="Metric",
                y="Score",
                color="ML Model Name",
                barmode="group",
                color_discrete_sequence=["#1F6F5B", "#C4A35A", "#4A7C9B", "#B85C38", "#5C4D7D", "#8C5A3C"],
            )
            bars.update_yaxes(range=[0, 1])
            bars.update_layout(height=420, legend_title="", margin=dict(t=20))
            st.plotly_chart(bars, width="stretch")
    else:
        preview = student_frame.copy()
        preview.insert(0, "Predicted outcome", predicted_labels)
        st.subheader(f"Predicted outcomes · {model_name}")
        st.dataframe(preview.head(25), width="stretch", hide_index=True)
        counts = pd.Series(predicted_labels).value_counts().rename_axis("Outcome").reset_index(name="Students")
        st.bar_chart(counts.set_index("Outcome"))

    with st.expander("Expected CSV layout"):
        st.write(
            "The file must include all 36 training features. Include a `Target` column "
            "(Dropout / Enrolled / Graduate) to compute Accuracy, AUC, Precision, Recall, F1, and MCC. "
            "A ready-made hold-out file is `test_data.csv` in the repository."
        )
        st.caption("Feature columns: " + ", ".join(feature_cols))


if __name__ == "__main__":
    main()
