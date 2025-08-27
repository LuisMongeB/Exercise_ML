import os

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filename: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""

    data_dir = os.path.join(os.getcwd(), "data", "raw")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} does not exist.")

    file_path = os.path.join(data_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    return pd.read_csv(file_path)


def create_subset(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Create a subset of the DataFrame with specified columns."""

    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"The following columns are missing from the DataFrame: {missing_cols}"
        )

    return df[columns]


def rename_columns(df: pd.DataFrame, new_column_names: list) -> pd.DataFrame:
    """Rename DataFrame columns."""

    if len(new_column_names) != len(df.columns):
        raise ValueError(
            "The number of new column names must match the number of existing columns."
        )

    df.columns = new_column_names
    return df


def split_dataset(
    df: pd.DataFrame, target_column: str, test_size=0.3, random_state=506
):
    """Split the dataset into training and testing sets."""

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X, y, X_train, X_test, y_train, y_test


def preprocess_dataset(
    filename: str, subset_columns: list, new_column_names: list, target_column: str
):
    """Complete preprocessing pipeline: load data, create subset, rename columns, and split dataset."""

    df = load_data(filename)
    subset_df = create_subset(df, subset_columns)
    renamed_df = rename_columns(subset_df, new_column_names)
    X, y, X_train, X_test, y_train, y_test = split_dataset(renamed_df, target_column)

    return X, y, X_train, X_test, y_train, y_test
