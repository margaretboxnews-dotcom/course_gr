import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from dataset_store import ChurnDatasetStore
from model_store import ModelStore


def _make_synthetic_df(n: int = 100) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "monthly_fee": 40.0 + (i % 50),
            "usage_hours": 50.0 + (i % 100),
            "support_requests": i % 6,
            "account_age_months": 6 + (i % 36),
            "failed_payments": i % 4,
            "region": ["europe", "asia", "america", "africa"][i % 4],
            "device_type": ["mobile", "desktop", "tablet"][i % 3],
            "payment_method": ["card", "paypal", "crypto"][i % 3],
            "autopay_enabled": i % 2,
            "churn": i % 2,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_csv(tmp_path):
    csv_path = tmp_path / "churn_dataset.csv"
    _make_synthetic_df().to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def client(synthetic_csv, monkeypatch):
    store = ChurnDatasetStore(csv_path=synthetic_csv)
    fresh_ms = ModelStore()
    monkeypatch.setattr(main, "dataset_store", store)
    monkeypatch.setattr(main, "model_store", fresh_ms)
    monkeypatch.setattr(fresh_ms, "load", lambda: False)
    monkeypatch.setattr(fresh_ms, "save", lambda: None)
    monkeypatch.setattr(fresh_ms, "_save_history", lambda: None)
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def trained_client(client):
    client.post("/model/train", json={})
    return client


VALID_FEATURES = {
    "monthly_fee": 45.5,
    "usage_hours": 120.0,
    "support_requests": 2,
    "account_age_months": 24,
    "failed_payments": 0,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1,
}
