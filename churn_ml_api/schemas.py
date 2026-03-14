from typing import Any

from pydantic import BaseModel, Field


class FeatureVectorChurn(BaseModel):
    monthly_fee: float = Field(..., examples=[45.5])
    usage_hours: float = Field(..., examples=[120.0])
    support_requests: int = Field(..., examples=[2])
    account_age_months: int = Field(..., examples=[24])
    failed_payments: int = Field(..., examples=[0])
    region: str = Field(..., examples=["europe"])
    device_type: str = Field(..., examples=["mobile"])
    payment_method: str = Field(..., examples=["card"])
    autopay_enabled: int = Field(..., examples=[1])


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class PredictionResponseChurn(BaseModel):
    churn: int = Field(..., description="Predicted class: 1 = churned, 0 = stayed")
    probability_churn: float = Field(..., description="Probability of churn (class 1)")
    probability_stay: float = Field(..., description="Probability of staying (class 0)")


class PredictBatchRequest(BaseModel):
    items: list[FeatureVectorChurn]


class TrainingConfigChurn(BaseModel):
    model_type: str = Field("logreg", description="Тип модели: logreg или random_forest", examples=["logreg"])
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Гиперпараметры модели. Переопределяют дефолты.",
        examples=[{"C": 0.5}],
    )


class ErrorResponse(BaseModel):
    code: str = Field(..., description="Machine-readable error code", examples=["model_not_trained"])
    message: str = Field(..., description="Human-readable description")
    details: Any = Field(None, description="Optional extra info (field errors, missing columns, etc.)")
