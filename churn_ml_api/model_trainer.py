from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset_store import CATEGORICAL_FEATURES, NUMERIC_FEATURES

SUPPORTED_MODELS: dict[str, type] = {
    "logreg": LogisticRegression,
    "random_forest": RandomForestClassifier,
}

DEFAULT_HYPERPARAMETERS: dict[str, dict] = {
    "logreg": {"max_iter": 1000, "random_state": 42},
    "random_forest": {"n_estimators": 100, "random_state": 42},
}


def build_pipeline(model_type: str = "logreg", hyperparameters: dict | None = None) -> Pipeline:
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Supported: {list(SUPPORTED_MODELS.keys())}"
        )
    params = {**DEFAULT_HYPERPARAMETERS[model_type], **(hyperparameters or {})}
    classifier = SUPPORTED_MODELS[model_type](**params)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_churn_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "logreg",
    hyperparameters: dict | None = None,
) -> tuple[Pipeline, dict]:
    pipeline = build_pipeline(model_type=model_type, hyperparameters=hyperparameters)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }
    return pipeline, metrics
