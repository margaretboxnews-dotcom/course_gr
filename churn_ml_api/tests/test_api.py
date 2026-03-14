from tests.conftest import VALID_FEATURES


# --- Health ---

def test_health_no_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["dataset_available"] is True
    assert body["trained_at"] is None


def test_health_after_train(trained_client):
    r = trained_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["dataset_available"] is True
    assert body["trained_at"] is not None


# --- Root ---

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["message"] == "ml churn service is running"


# --- /predict (single) ---

def test_predict_without_model_returns_409(client):
    r = client.post("/predict", json=VALID_FEATURES)
    assert r.status_code == 409
    assert r.json()["code"] == "model_not_trained"


def test_predict_validation_error_missing_field(client):
    r = client.post("/predict", json={"monthly_fee": 45.5})
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"


def test_predict_validation_error_wrong_type(client):
    bad = {**VALID_FEATURES, "monthly_fee": "not_a_number"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert r.json()["code"] == "validation_error"


def test_predict_after_train(trained_client):
    r = trained_client.post("/predict", json=VALID_FEATURES)
    assert r.status_code == 200
    body = r.json()
    assert body["churn"] in [0, 1]
    assert 0.0 <= body["probability_churn"] <= 1.0
    assert 0.0 <= body["probability_stay"] <= 1.0
    assert round(body["probability_churn"] + body["probability_stay"], 2) == 1.0


# --- /predict/batch ---

def test_predict_batch_empty_returns_400(client):
    r = client.post("/predict/batch", json={"items": []})
    assert r.status_code == 400
    assert r.json()["code"] == "empty_batch"


def test_predict_batch_after_train(trained_client):
    second = {**VALID_FEATURES, "monthly_fee": 80.0, "region": "asia"}
    r = trained_client.post("/predict/batch", json={"items": [VALID_FEATURES, second]})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    for item in body:
        assert item["churn"] in [0, 1]


# --- /model/train ---

def test_model_train_returns_metrics(client):
    r = client.post("/model/train", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "trained"
    assert {"accuracy", "f1", "roc_auc"}.issubset(body["metrics"].keys())
    assert body["train_rows"] > 0
    assert body["test_rows"] > 0


def test_model_train_custom_test_size(client):
    r = client.post("/model/train?test_size=0.3", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["test_rows"] == 30  # 30% of 100


def test_model_train_random_forest(client):
    r = client.post("/model/train", json={"model_type": "random_forest"})
    assert r.status_code == 200
    assert r.json()["status"] == "trained"


def test_model_train_random_forest_with_hyperparams(client):
    r = client.post("/model/train", json={"model_type": "random_forest", "hyperparameters": {"n_estimators": 10}})
    assert r.status_code == 200


def test_model_train_logreg_with_hyperparams(client):
    r = client.post("/model/train", json={"model_type": "logreg", "hyperparameters": {"C": 0.5}})
    assert r.status_code == 200


def test_model_train_invalid_model_type(client):
    r = client.post("/model/train", json={"model_type": "xgboost"})
    assert r.status_code == 400
    assert r.json()["code"] == "invalid_model_type"


# --- /model/status ---

def test_model_status_not_trained(client):
    r = client.get("/model/status")
    assert r.status_code == 200
    assert r.json()["is_trained"] is False


def test_model_status_after_train(trained_client):
    r = trained_client.get("/model/status")
    assert r.status_code == 200
    body = r.json()
    assert body["is_trained"] is True
    assert body["trained_at"] is not None
    assert body["metrics"] is not None


# --- /model/metrics ---

def test_model_metrics_empty_history(client):
    r = client.get("/model/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["history"] == []
    assert body["latest"] is None


def test_model_metrics_after_train(trained_client):
    r = trained_client.get("/model/metrics")
    assert r.status_code == 200
    body = r.json()
    assert len(body["history"]) == 1
    record = body["history"][0]
    assert record["model_type"] == "LogisticRegression"
    assert {"accuracy", "f1", "roc_auc"}.issubset(record["metrics"].keys())
    assert record["train_rows"] > 0


# --- /dataset/info ---

def test_dataset_info(client):
    r = client.get("/dataset/info")
    assert r.status_code == 200
    body = r.json()
    assert body["rows"] == 100


# --- /dataset/preview ---

def test_dataset_preview_default(client):
    r = client.get("/dataset/preview")
    assert r.status_code == 200
    assert len(r.json()["rows"]) == 5


def test_dataset_preview_custom_n(client):
    r = client.get("/dataset/preview?n=10")
    assert r.status_code == 200
    assert len(r.json()["rows"]) == 10


# --- /dataset/split-info ---

def test_dataset_split_info(client):
    r = client.get("/dataset/split-info")
    assert r.status_code == 200
    body = r.json()
    assert body["train_rows"] + body["test_rows"] == 100
