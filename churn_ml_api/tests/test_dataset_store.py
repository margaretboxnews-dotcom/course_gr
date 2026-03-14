import pandas as pd
import pytest

from dataset_store import ChurnDatasetStore


def test_load_df(synthetic_csv):
    store = ChurnDatasetStore(csv_path=synthetic_csv)
    df = store.load_df()
    assert len(df) == 100
    assert "churn" in df.columns


def test_load_df_cached(synthetic_csv):
    store = ChurnDatasetStore(csv_path=synthetic_csv)
    df1 = store.load_df()
    df2 = store.load_df()
    assert df1 is df2


def test_file_not_found(tmp_path):
    store = ChurnDatasetStore(csv_path=tmp_path / "nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        store.load_df()


def test_validate_columns_ok(synthetic_csv):
    store = ChurnDatasetStore(csv_path=synthetic_csv)
    store.validate_columns()  # should not raise


def test_validate_columns_missing(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"monthly_fee": [1.0], "churn": [0]}).to_csv(bad_csv, index=False)
    store = ChurnDatasetStore(csv_path=bad_csv)
    with pytest.raises(ValueError, match="missing columns"):
        store.validate_columns()


def test_prepare_xy_shape(synthetic_csv):
    store = ChurnDatasetStore(csv_path=synthetic_csv)
    X, y = store.prepare_xy()
    assert X.shape == (100, 9)
    assert len(y) == 100


def test_prepare_xy_target_values(synthetic_csv):
    store = ChurnDatasetStore(csv_path=synthetic_csv)
    _, y = store.prepare_xy()
    assert set(y.unique()).issubset({0, 1})


def test_preview_count(synthetic_csv):
    store = ChurnDatasetStore(csv_path=synthetic_csv)
    rows = store.preview(5)
    assert len(rows) == 5
    assert "monthly_fee" in rows[0]


def test_info_structure(synthetic_csv):
    store = ChurnDatasetStore(csv_path=synthetic_csv)
    info = store.info()
    assert info["rows"] == 100
    assert "churn_distribution" in info
    assert "numeric_features" in info
    assert "categorical_features" in info
