import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from validation import validate_data


SEED = 42
MINIMUM_AUC = 0.60
TEST_SIZE = 0.20
MAX_ITER = 1000

NUMERIC_FEATURES = [
    "tenure",
    "monthly_charges",
    "support_calls",
]

CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
]

TARGET = "churn"


def build_model():
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
    ])

    return Pipeline([
        ("preprocess", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=MAX_ITER,
                random_state=SEED,
            ),
        ),
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Train the customer churn prediction model"
    )
    parser.add_argument(
        "--check-quality-gate",
        action="store_true",
        help="Fail if ROC-AUC is below 0.60",
    )
    args = parser.parse_args()

    # Validate dataset
    df = validate_data()

    if df.empty:
        raise ValueError("No valid data available for training")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    # Check target classes
    if y.nunique() != 2:
        raise ValueError(
            "Training requires both churn classes: 0 and 1"
        )

    # Ensure enough rows for a stratified split
    class_counts = y.value_counts()

    if class_counts.min() < 2:
        raise ValueError(
            "Each churn class must have at least 2 rows "
            "for a stratified train/test split"
        )

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=y,
        )
    except ValueError as exc:
        raise ValueError(
            f"Unable to split the dataset: {exc}. "
            "Add more labeled examples."
        ) from exc

    # Create output directory
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure MLflow
    mlflow.set_experiment("Customer Churn Prediction")

    with mlflow.start_run() as run:
        model = build_model()

        # Train model
        model.fit(X_train, y_train)

        # Evaluate model
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities)

        # Log parameters
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "random_seed": SEED,
            "test_size": TEST_SIZE,
            "max_iter": MAX_ITER,
            "dataset_rows": len(df),
            "training_rows": len(X_train),
            "test_rows": len(X_test),
        })

        # Log metrics
        mlflow.log_metrics({
            "accuracy": float(accuracy),
            "auc": float(auc),
        })

        # Save model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
        )

        # Save local model
        joblib.dump(model, output_dir / "model.joblib")

        # Save evaluation metrics
        metrics = {
            "accuracy": float(accuracy),
            "auc": float(auc),
            "minimum_auc": MINIMUM_AUC,
            "run_id": run.info.run_id,
            "rows": int(len(df)),
        }

        with open(
            output_dir / "metrics.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metrics, file, indent=4)

        print("\nModel training completed")
        print(f"Dataset rows: {len(df)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"MLflow run ID: {run.info.run_id}")
        print(f"Model saved to: {output_dir / 'model.joblib'}")
        print(f"Metrics saved to: {output_dir / 'metrics.json'}")

        # Quality gate
        if args.check_quality_gate and auc < MINIMUM_AUC:
            raise RuntimeError(
                f"Quality gate failed: ROC-AUC {auc:.4f} "
                f"is below {MINIMUM_AUC:.2f}"
            )

        if auc >= MINIMUM_AUC:
            print("Quality gate passed.")
        else:
            print(
                "Quality gate not checked. "
                "Use --check-quality-gate to enforce it."
            )


if __name__ == "__main__":
    main()
