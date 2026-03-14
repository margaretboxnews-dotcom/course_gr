import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sklearn.model_selection import train_test_split

from dataset_store import FEATURE_COLUMNS, ChurnDatasetStore
from model_store import model_store
from model_trainer import train_churn_model
from schemas import (
    ErrorResponse,
    FeatureVectorChurn,
    PredictBatchRequest,
    PredictionResponseChurn,
    TrainingConfigChurn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent


def resolve_dataset_path() -> Path:
    candidates = [
        PROJECT_DIR / "data" / "churn_dataset.csv",
        PROJECT_DIR / "churn_dataset.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


dataset_store = ChurnDatasetStore(csv_path=resolve_dataset_path())


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = model_store.load()
    if loaded:
        logger.info("Model loaded from disk (trained at %s)", model_store.trained_at)
    else:
        logger.info("No saved model found, starting fresh")
    yield


app = FastAPI(
    title="Churn ML API",
    lifespan=lifespan,
    responses={
        400: {"model": ErrorResponse, "description": "Bad request (invalid data, empty dataset, etc.)"},
        404: {"model": ErrorResponse, "description": "Resource not found (dataset missing, etc.)"},
        409: {"model": ErrorResponse, "description": "Conflict (model not trained yet)"},
        422: {"model": ErrorResponse, "description": "Validation error (wrong field types or missing fields)"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(code=code, message=message, details=details).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code_map = {
        400: "bad_request",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        500: "internal_error",
    }
    code = code_map.get(exc.status_code, "error")
    if exc.status_code >= 500:
        logger.error("HTTP %s %s → %s", exc.status_code, request.url.path, exc.detail)
    elif exc.status_code >= 400:
        logger.warning("HTTP %s %s → %s", exc.status_code, request.url.path, exc.detail)
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(**exc.detail).model_dump(),
        )
    return _error(exc.status_code, code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    details = [
        {"field": " -> ".join(str(loc) for loc in e["loc"]), "problem": e["msg"]}
        for e in exc.errors()
    ]
    logger.warning("Validation error on %s: %s", request.url.path, details)
    return _error(422, "validation_error", "Request body contains invalid or missing fields", details)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return _error(500, "internal_error", f"Unexpected error: {type(exc).__name__}: {exc}")


@app.get("/")
def root():
    return {"message": "ml churn service is running"}


@app.get("/health")
def health():
    """Состояние сервиса: доступность модели и датасета."""
    dataset_ok = False
    try:
        dataset_store.load_df()
        dataset_ok = True
    except Exception:
        pass

    status = "ok" if model_store.is_trained and dataset_ok else "degraded"
    return {
        "status": status,
        "model_loaded": model_store.is_trained,
        "dataset_available": dataset_ok,
        "trained_at": model_store.trained_at.isoformat() if model_store.trained_at else None,
    }


def _run_predict(items: list[FeatureVectorChurn]) -> list[PredictionResponseChurn]:
    if not model_store.is_trained:
        raise HTTPException(
            status_code=409,
            detail={"code": "model_not_trained", "message": "Model is not trained yet. Call POST /model/train first.", "details": None},
        )
    df = pd.DataFrame([item.model_dump() for item in items])[FEATURE_COLUMNS]
    predictions = model_store.pipeline.predict(df)
    probabilities = model_store.pipeline.predict_proba(df)
    results = [
        PredictionResponseChurn(
            churn=int(predictions[i]),
            probability_churn=round(float(probabilities[i][1]), 4),
            probability_stay=round(float(probabilities[i][0]), 4),
        )
        for i in range(len(items))
    ]
    logger.info("Predicted churn for %d item(s)", len(items))
    return results


@app.post(
    "/predict",
    response_model=PredictionResponseChurn,
    responses={
        409: {"model": ErrorResponse, "description": "Model not trained", "content": {"application/json": {"example": {"code": "model_not_trained", "message": "Model is not trained yet. Call POST /model/train first.", "details": None}}}},
        422: {"model": ErrorResponse, "description": "Invalid input", "content": {"application/json": {"example": {"code": "validation_error", "message": "Request body contains invalid or missing fields", "details": [{"field": "body -> monthly_fee", "problem": "Input should be a valid number"}]}}}},
    },
)
def predict_churn(features: FeatureVectorChurn):
    """Предсказание churn для одного клиента."""
    return _run_predict([features])[0]


@app.post("/predict/batch", response_model=list[PredictionResponseChurn])
def predict_churn_batch(request: PredictBatchRequest):
    """Предсказание churn для списка клиентов."""
    if not request.items:
        raise HTTPException(
            status_code=400,
            detail={"code": "empty_batch", "message": "items list must not be empty", "details": None},
        )
    return _run_predict(request.items)


@app.get("/dataset/preview")
def dataset_preview(n: int = Query(5, ge=1, le=200)):
    try:
        return {"rows": dataset_store.preview(n)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "dataset_not_found", "message": str(e), "details": None})
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail={"code": "dataset_error", "message": str(e), "details": None})


@app.get("/dataset/info")
def dataset_info():
    try:
        return dataset_store.info()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "dataset_not_found", "message": str(e), "details": None})
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail={"code": "dataset_error", "message": str(e), "details": None})


@app.get("/dataset/split-info")
def dataset_split_info(test_size: float = Query(0.2, ge=0.05, le=0.5)):
    try:
        return dataset_store.train_test_split_info(test_size=test_size)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "dataset_not_found", "message": str(e), "details": None})
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail={"code": "dataset_error", "message": str(e), "details": None})


@app.post(
    "/model/train",
    responses={
        404: {"model": ErrorResponse, "description": "Dataset file not found", "content": {"application/json": {"example": {"code": "dataset_not_found", "message": "Dataset CSV not found: .../churn_dataset.csv", "details": None}}}},
        400: {"model": ErrorResponse, "description": "Dataset is empty or invalid", "content": {"application/json": {"example": {"code": "dataset_empty", "message": "Dataset is empty", "details": None}}}},
    },
)
def model_train(
    config: TrainingConfigChurn,
    test_size: float = Query(0.2, ge=0.05, le=0.5),
):
    """Обучить модель на churn_dataset.csv. model_type: logreg | random_forest."""
    logger.info("Training started (model=%s, test_size=%.2f)", config.model_type, test_size)
    try:
        X, y = dataset_store.prepare_xy()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail={"code": "dataset_not_found", "message": str(e), "details": None})
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail={"code": "dataset_error", "message": str(e), "details": None})

    if X.empty:
        raise HTTPException(status_code=400, detail={"code": "dataset_empty", "message": "Dataset is empty", "details": None})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    try:
        pipeline, metrics = train_churn_model(
            X_train, y_train, X_test, y_test,
            model_type=config.model_type,
            hyperparameters=config.hyperparameters,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"code": "invalid_model_type", "message": str(e), "details": None})
    model_store.update(pipeline, metrics, train_rows=len(X_train), test_rows=len(X_test))
    logger.info("Training complete (model=%s): %s", config.model_type, metrics)

    return {
        "status": "trained",
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "metrics": metrics,
    }


@app.get("/model/status")
def model_status():
    return model_store.status()


@app.get("/model/metrics")
def model_metrics(
    limit: int = Query(10, ge=1, le=100),
    model_type: str | None = Query(None),
):
    """История обучений модели с метриками."""
    history = model_store.get_history(model_type=model_type, limit=limit)
    return {
        "latest": history[0] if history else None,
        "history": history,
    }


@app.get("/model/schema")
def model_schema():
    """Описание признаков: порядок, типы и допустимые значения категориальных."""
    return dataset_store.schema()
