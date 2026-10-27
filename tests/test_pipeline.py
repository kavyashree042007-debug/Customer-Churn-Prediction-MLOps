
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from validation import validate_data


DATA_PATH = Path("data/customers.csv")
MODEL_PATH = Path("artifacts/model.joblib")
METRICS_PATH = Path("artifacts/metrics.json")


def test_data_is_valid():
    df = validate_data(DATA_PATH)

    assert len(df) > 0
    assert "churn" in df.columns
    assert df["churn"].isin([0, 1]).all()


def test_missing_values_are_removed(tmp_path):
    # Create a test dataset with a missing value
    df = pd.DataFrame({
        "tenure": [12, np.nan, 36, 48],
        "monthly_charges": [50.0, 60.0, 70.0, 80.0],
        "support_calls": [1, 2, 0, 3],
        "contract": [
            "Monthly", "One Year", "Two Year", "Monthly"
        ],
        "internet_service": [
            "DSL", "Fiber", "None", "DSL"
        ],
        "churn": [0, 1, 0, 1],
    })

    bad_file = tmp_path / "bad_customers.csv"
    df.to_csv(bad_file, index=False)

    cleaned_df = validate_data(bad_file)

    # The incomplete row should be removed
    assert len(cleaned_df) == 3
    assert not cleaned_df.isnull().any().any()


def test_missing_file_raises_error(tmp_path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        validate_data(missing_file)


def test_model_and_metrics_exist():
    assert MODEL_PATH.exists(), (
        "Model not found. Run python train.py first."
    )
    assert METRICS_PATH.exists(), (
        "Metrics not found. Run python train.py first."
    )


def test_model_prediction_shape():
    if not MODEL_PATH.exists():
        pytest.skip("Run python train.py before testing the model")

    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame([{
        "tenure": 12,
        "monthly_charges": 80.0,
        "support_calls": 3,
        "contract": "Monthly",
        "internet_service": "Fiber",
    }])

    prediction = model.predict(sample)
    probability = model.predict_proba(sample)

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]
    assert probability.shape == (1, 2)
    assert np.allclose(probability.sum(axis=1), 1.0)


def test_model_quality():
    if not METRICS_PATH.exists():
        pytest.skip("Run python train.py before testing metrics")

    with open(METRICS_PATH, encoding="utf-8") as file:
        metrics = json.load(file)

    assert 0.0 <= metrics["auc"] <= 1.0
    assert metrics["minimum_auc"] == 0.60
    assert metrics["auc"] >= metrics["minimum_auc"], (
        f"ROC-AUC {metrics['auc']:.4f} is below "
        f"the required {metrics['minimum_auc']:.2f}"
    )
