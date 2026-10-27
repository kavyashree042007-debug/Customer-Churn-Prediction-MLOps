
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/customers.csv")

REQUIRED_COLUMNS = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract",
    "internet_service",
    "churn",
]


def validate_data(file_path=DATA_PATH):
    file_path = Path(file_path)

    # Check whether the dataset exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path.resolve()}"
        )

    # Load dataset
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Dataset is empty")

    # Check required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # Convert numeric columns; invalid text raises an error
    numeric_columns = [
        "tenure",
        "monthly_charges",
        "support_calls",
        "churn",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        )

    # Detect missing values, including blank strings
    df[REQUIRED_COLUMNS] = df[REQUIRED_COLUMNS].replace(
        r"^\s*$", float("nan"), regex=True
    )

    missing_values = df[REQUIRED_COLUMNS].isnull().sum()

    if missing_values.any():
        print("Missing values found:")
        print(missing_values[missing_values > 0])

        original_rows = len(df)

        # Remove incomplete rows
        df = df.dropna(subset=REQUIRED_COLUMNS).copy()

        removed_rows = original_rows - len(df)

        print(f"Rows removed: {removed_rows}")
        print(f"Rows remaining: {len(df)}")

    if df.empty:
        raise ValueError(
            "No rows remain after removing missing values"
        )

    # Validate churn values
    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn must contain only 0 and 1")

    if df["churn"].nunique() != 2:
        raise ValueError(
            "Target must contain both churn classes (0 and 1)"
        )

    # Validate non-negative numeric features
    for column in [
        "tenure",
        "monthly_charges",
        "support_calls",
    ]:
        if (df[column] < 0).any():
            raise ValueError(
                f"{column} cannot contain negative values"
            )

    # Validate contract categories
    allowed_contracts = {
        "Monthly",
        "One Year",
        "Two Year",
    }

    actual_contracts = set(df["contract"].unique())

    if not actual_contracts.issubset(allowed_contracts):
        raise ValueError(
            f"Unexpected contract types: "
            f"{sorted(actual_contracts - allowed_contracts)}"
        )

    # Validate internet service categories
    allowed_services = {
        "DSL",
        "Fiber",
        "None",
    }

    actual_services = set(df["internet_service"].unique())

    if not actual_services.issubset(allowed_services):
        raise ValueError(
            f"Unexpected internet services: "
            f"{sorted(actual_services - allowed_services)}"
        )

    print(f"Data validation passed: {len(df)} rows")

    return df


if __name__ == "__main__":
    validate_data()
