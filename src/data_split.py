from datetime import datetime
from typing import Tuple

import pandas as pd


def train_test_split(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column_name: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split data into train and test sets based on a cutoff date"""
    # First, ensure pickup_hour is in datetime format
    df_copy = df.copy()
    if df_copy['pickup_hour'].dtype == 'object':  # If it's a string
        df_copy['pickup_hour'] = pd.to_datetime(df_copy['pickup_hour'])

    # Now perform the split
    train_data = df_copy[df_copy.pickup_hour < cutoff_date].reset_index(drop=True)
    test_data = df_copy[df_copy.pickup_hour >= cutoff_date].reset_index(drop=True)

    X_train = train_data.drop(columns=[target_column_name])
    y_train = train_data[target_column_name]
    X_test = test_data.drop(columns=[target_column_name])
    y_test = test_data[target_column_name]

    return X_train, y_train, X_test, y_test