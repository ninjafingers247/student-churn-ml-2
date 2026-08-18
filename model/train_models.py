"""
Train six classification models on the UCI Students' Dropout dataset.

Run from the project root:
    python model/train_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "data.csv"
MODEL_DIR = ROOT / "model"
TEST_DATA_PATH = ROOT / "test_data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_FILE_MAP = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "XGBoost": "xgboost.joblib",
}


def tidy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [str(col).replace("\t", "").strip() for col in cleaned.columns]
    return cleaned


def load_enrolment_table(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, sep=";")
    return tidy_columns(raw)


def score_classifier(estimator, features, true_labels) -> dict:
    predicted = estimator.predict(features)
    probabilities = estimator.predict_proba(features)
    return {
        "Accuracy": round(float(accuracy_score(true_labels, predicted)), 4),
        "AUC": round(
            float(
                roc_auc_score(
                    true_labels,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )
            ),
            4,
        ),
        "Precision": round(
            float(precision_score(true_labels, predicted, average="weighted", zero_division=0)),
            4,
        ),
        "Recall": round(
            float(recall_score(true_labels, predicted, average="weighted", zero_division=0)),
            4,
        ),
        "F1": round(
            float(f1_score(true_labels, predicted, average="weighted", zero_division=0)),
            4,
        ),
        "MCC": round(float(matthews_corrcoef(true_labels, predicted)), 4),
    }


def build_model_zoo() -> dict:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2500,
                        C=0.75,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=11,
            min_samples_split=18,
            min_samples_leaf=7,
            criterion="gini",
            random_state=RANDOM_STATE,
        ),
        "kNN": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=9,
                        weights="distance",
                        p=2,
                    ),
                ),
            ]
        ),
        "Naive Bayes": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", GaussianNB()),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=14,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=220,
            max_depth=5,
            learning_rate=0.07,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_weight=3,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    enrolment = load_enrolment_table(DATA_PATH)
    outcome_col = "Target"
    feature_cols = [col for col in enrolment.columns if col != outcome_col]

    predictors = enrolment[feature_cols]
    outcome_text = enrolment[outcome_col]

    label_codec = LabelEncoder()
    outcome_codes = label_codec.fit_transform(outcome_text)

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        predictors,
        outcome_codes,
        test_size=TEST_SIZE,
        stratify=outcome_codes,
        random_state=RANDOM_STATE,
    )

    holdout_export = x_holdout.copy()
    holdout_export[outcome_col] = label_codec.inverse_transform(y_holdout)
    holdout_export.to_csv(TEST_DATA_PATH, index=False)

    metrics_board = {}
    for model_name, estimator in build_model_zoo().items():
        estimator.fit(x_train, y_train)
        metrics_board[model_name] = score_classifier(estimator, x_holdout, y_holdout)
        joblib.dump(estimator, MODEL_DIR / MODEL_FILE_MAP[model_name])
        print(f"Saved {model_name}: {metrics_board[model_name]}")

    preprocessing = {
        "feature_columns": feature_cols,
        "class_names": list(label_codec.classes_),
        "label_encoder": label_codec,
    }
    joblib.dump(preprocessing, MODEL_DIR / "preprocessing.joblib")

    with (MODEL_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_board, handle, indent=2)

    print("\nComparison table (hold-out test set)")
    header = f"{'ML Model Name':<22} {'Accuracy':>10} {'AUC':>8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'MCC':>8}"
    print(header)
    print("-" * len(header))
    for name, scores in metrics_board.items():
        print(
            f"{name:<22} {scores['Accuracy']:>10.4f} {scores['AUC']:>8.4f} "
            f"{scores['Precision']:>10.4f} {scores['Recall']:>8.4f} "
            f"{scores['F1']:>8.4f} {scores['MCC']:>8.4f}"
        )

    winner = max(metrics_board.items(), key=lambda item: item[1]["F1"])
    print(f"\nHighest F1 on this dataset: {winner[0]}")
    print(f"Wrote hold-out CSV -> {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
