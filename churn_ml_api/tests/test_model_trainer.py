from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from dataset_store import ChurnDatasetStore
from model_trainer import build_pipeline, train_churn_model


def test_build_pipeline_structure():
    pipeline = build_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_train_returns_pipeline_and_metrics(synthetic_csv):
    store = ChurnDatasetStore(csv_path=synthetic_csv)
    X, y = store.prepare_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline, metrics = train_churn_model(X_train, y_train, X_test, y_test)

    assert isinstance(pipeline, Pipeline)
    assert set(metrics.keys()) == {"accuracy", "f1", "roc_auc"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_trained_pipeline_predicts(synthetic_csv):
    store = ChurnDatasetStore(csv_path=synthetic_csv)
    X, y = store.prepare_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline, _ = train_churn_model(X_train, y_train, X_test, y_test)

    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1})
