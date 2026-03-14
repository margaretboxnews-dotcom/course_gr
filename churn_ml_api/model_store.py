from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

MODEL_PATH = Path(__file__).resolve().parent / "models" / "churn_model.joblib"
HISTORY_PATH = Path(__file__).resolve().parent / "models" / "training_history.json"


@dataclass
class ModelStore:
    pipeline: Pipeline | None = None
    trained_at: datetime | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    model_type: str | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    def save(self) -> None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "trained_at": self.trained_at, "metrics": self.metrics},
            MODEL_PATH,
        )

    def _save_history(self) -> None:
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_PATH.write_text(json.dumps(self.history, indent=2, default=str))

    def _load_history(self) -> None:
        if HISTORY_PATH.exists():
            self.history = json.loads(HISTORY_PATH.read_text())

    def load(self) -> bool:
        """Загружает модель и историю с диска. Возвращает True если модель найдена."""
        self._load_history()
        if not MODEL_PATH.exists():
            return False
        data = joblib.load(MODEL_PATH)
        self.pipeline = data["pipeline"]
        self.trained_at = data["trained_at"]
        self.metrics = data["metrics"]
        return True

    def update(self, pipeline: Pipeline, metrics: dict[str, Any], train_rows: int, test_rows: int) -> None:
        self.pipeline = pipeline
        self.trained_at = datetime.now(timezone.utc)
        self.metrics = metrics
        self.save()

        classifier = pipeline.named_steps["classifier"]
        self.model_type = type(classifier).__name__
        self.hyperparameters = classifier.get_params()

        record: dict[str, Any] = {
            "id": len(self.history) + 1,
            "trained_at": self.trained_at.isoformat(),
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "metrics": metrics,
            "train_rows": train_rows,
            "test_rows": test_rows,
        }
        self.history.append(record)
        self._save_history()

    def get_history(self, model_type: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        records = self.history
        if model_type:
            records = [r for r in records if r.get("model_type") == model_type]
        if limit:
            records = records[-limit:]
        return list(reversed(records))

    def status(self) -> dict[str, Any]:
        return {
            "is_trained": self.is_trained,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "model_path": str(MODEL_PATH) if self.is_trained else None,
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters if self.is_trained else None,
            "metrics": self.metrics if self.is_trained else None,
        }


model_store = ModelStore()
