# Churn ML API

FastAPI service for customer churn prediction. Trains a classification model on tabular data and exposes a REST API for predictions.

## Project Structure

```
├── main.py            # FastAPI app and all endpoints
├── schemas.py         # Pydantic models (requests and responses)
├── dataset_store.py   # Dataset loading and preprocessing
├── model_trainer.py   # sklearn Pipeline building and training
├── model_store.py     # Model persistence (joblib) and training history
├── tests/
│   ├── conftest.py    # pytest fixtures (synthetic data, TestClient)
│   ├── test_api.py    # Integration tests for all endpoints
│   ├── test_dataset_store.py
│   └── test_model_trainer.py
├── Dockerfile
├── requirements.txt
└── churn_dataset.csv  # Dataset
```

## Dataset

`churn_dataset.csv` contains 2000 rows with the following features:

| Field | Type | Description |
|-------|------|-------------|
| `monthly_fee` | float | Monthly subscription cost |
| `usage_hours` | float | Hours of usage per month |
| `support_requests` | int | Number of support tickets |
| `account_age_months` | int | Account age in months |
| `failed_payments` | int | Number of failed payments |
| `region` | str | europe / asia / america / africa |
| `device_type` | str | mobile / desktop / tablet |
| `payment_method` | str | card / paypal / crypto |
| `autopay_enabled` | int | 0 or 1 |
| `churn` | int | **Target**: 1 = churned, 0 = stayed |

## Run Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --reload
```

Swagger UI available at: http://localhost:8000/docs

## Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Example Requests

### Train a model (LogisticRegression by default)

```bash
curl -X POST http://localhost:8000/model/train \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Train RandomForest with custom hyperparameters

```bash
curl -X POST "http://localhost:8000/model/train?test_size=0.2" \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "hyperparameters": {"n_estimators": 200, "max_depth": 5}}'
```

Response:
```json
{
  "status": "trained",
  "train_rows": 1600,
  "test_rows": 400,
  "metrics": {"accuracy": 0.89, "f1": 0.72, "roc_auc": 0.94}
}
```

### Single prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "monthly_fee": 80.0,
    "usage_hours": 30.0,
    "support_requests": 5,
    "account_age_months": 6,
    "failed_payments": 3,
    "region": "asia",
    "device_type": "mobile",
    "payment_method": "paypal",
    "autopay_enabled": 0
  }'
```

Response:
```json
{"churn": 1, "probability_churn": 0.8231, "probability_stay": 0.1769}
```

### Batch prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"items": [
    {"monthly_fee": 45.5, "usage_hours": 120.0, "support_requests": 1,
     "account_age_months": 36, "failed_payments": 0, "region": "europe",
     "device_type": "desktop", "payment_method": "card", "autopay_enabled": 1},
    {"monthly_fee": 80.0, "usage_hours": 20.0, "support_requests": 7,
     "account_age_months": 3, "failed_payments": 4, "region": "africa",
     "device_type": "mobile", "payment_method": "crypto", "autopay_enabled": 0}
  ]}'
```

### Model status and training history

```bash
curl http://localhost:8000/model/status
curl http://localhost:8000/model/metrics
curl http://localhost:8000/health
```

## Run Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service status |
| GET | `/health` | Health check: model + dataset |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch prediction |
| POST | `/model/train` | Train the model |
| GET | `/model/status` | Model status and parameters |
| GET | `/model/metrics` | Training history |
| GET | `/model/schema` | Feature schema |
| GET | `/dataset/info` | Dataset info |
| GET | `/dataset/preview` | First N rows of dataset |
| GET | `/dataset/split-info` | Train/test split breakdown |
| GET | `/docs` | Swagger UI |
