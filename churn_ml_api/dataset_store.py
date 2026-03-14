from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from schemas import DatasetRowChurn, FeatureVectorChurn


NUMERIC_FEATURES: list[str] = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_FEATURES: list[str] = [
    "region",
    "device_type",
    "payment_method",
]

FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class ChurnDatasetStore:
    csv_path: Path
    _df: pd.DataFrame | None = None

    def load_df(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        self._df = df
        return df

    def required_columns(self) -> list[str]:
        return list(DatasetRowChurn.model_fields.keys())

    def feature_columns(self) -> list[str]:
        return list(FEATURE_COLUMNS)

    def validate_columns(self) -> None:
        df = self.load_df()
        missing = [c for c in self.required_columns() if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing columns: {missing}")

    def preview(self, n: int) -> list[dict[str, Any]]:
        self.validate_columns()
        df = self.load_df().head(n)
        records: list[dict[str, Any]] = df.to_dict(orient="records")
        validated = [DatasetRowChurn.model_validate(r).model_dump() for r in records]
        return validated

    def info(self) -> dict[str, Any]:
        self.validate_columns()
        df = self.load_df()
        churn_counts = df["churn"].value_counts(dropna=False).to_dict()
        churn_distribution = [
            {"class": int(k) if pd.notna(k) else None, "count": int(v)}
            for k, v in churn_counts.items()
        ]
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "feature_names": self.feature_columns(),
            "churn_distribution": churn_distribution,
        }

    def schema(self) -> dict[str, Any]:
        """Возвращает описание признаков: типы и допустимые значения для категориальных."""
        type_map = {
            "monthly_fee": "float",
            "usage_hours": "float",
            "support_requests": "int",
            "account_age_months": "int",
            "failed_payments": "int",
            "autopay_enabled": "int (0 or 1)",
            "region": "str",
            "device_type": "str",
            "payment_method": "str",
        }

        allowed: dict[str, list[str]] = {}
        try:
            df = self.load_df()
            for col in CATEGORICAL_FEATURES:
                allowed[col] = sorted(df[col].dropna().unique().tolist())
        except FileNotFoundError:
            for col in CATEGORICAL_FEATURES:
                allowed[col] = []

        features = []
        for col in FEATURE_COLUMNS:
            entry: dict[str, Any] = {"name": col, "type": type_map.get(col, "unknown")}
            if col in CATEGORICAL_FEATURES:
                entry["allowed_values"] = allowed[col]
            features.append(entry)

        return {
            "feature_count": len(FEATURE_COLUMNS),
            "feature_order": FEATURE_COLUMNS,
            "features": features,
        }

    def prepare_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Отделяет X и y, обрабатывает пропуски и выбирает нужные столбцы."""
        self.validate_columns()
        df = self.load_df().copy()

        for col in NUMERIC_FEATURES:
            if col in df.columns:
                median = df[col].median()
                df[col] = df[col].fillna(median)
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                mode = df[col].mode(dropna=True)
                if not mode.empty:
                    df[col] = df[col].fillna(mode.iloc[0])

        feature_cols = self.feature_columns()
        X = df[feature_cols]
        y = df["churn"]
        return X, y

    def train_test_split_info(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Разбивает данные на train/test и считает распределение churn."""
        X, y = self.prepare_xy()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        def _dist(series: pd.Series) -> list[dict[str, int]]:
            counts = series.value_counts(dropna=False).to_dict()
            return [
                {"class": int(k) if pd.notna(k) else None, "count": int(v)}
                for k, v in counts.items()
            ]

        return {
            "test_size": test_size,
            "random_state": random_state,
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
            "train_churn_distribution": _dist(y_train),
            "test_churn_distribution": _dist(y_test),
        }

