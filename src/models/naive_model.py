# models/naive_model.py

import numpy as np
from utils import evaluate_forecast, measure_time

def naive_forecast(X):
    """
    Implements the naive method for forecasting.
    
    Parameters:
    X (ndarray): The input time series.
    
    Returns:
    ndarray: The naive forecasts, which are simply the last observed value.
    """
    return X[:, -1]

@measure_time
def train_and_test_naive_model(X_train, y_train, X_test, y_test, minmax_scaler):
    """
    Trains and tests the naive forecasting model.
    
    Parameters:
    X_train (ndarray): Training input time series.
    y_train (ndarray): Training output time series (not used in naive method).
    X_test (ndarray): Testing input time series.
    y_test (ndarray): Testing output time series.
    minmax_scaler (object): Scaler used for normalizing/denormalizing data.
    
    Returns:
    dict: Evaluation metrics for the naive forecasts on the test set.
    """
    # Generate predictions using the naive method
    y_test_pred = naive_forecast(X_test)

    # Evaluate the predictions using the provided evaluation function
    test_evaluation = evaluate_forecast(y_test, y_test_pred, minmax_scaler)

    return test_evaluation
