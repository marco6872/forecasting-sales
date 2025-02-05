# models/naive_model.py

import numpy as np
from utils import evaluate_forecast, measure_time


def naive_forecast(X):
    """
    Implements the naive method for forecasting.
    
    Parameters:
    X (ndarray): The input time series.
    
    Returns:
    ndarray: The naive forecasts.
    """
    return np.tile(X[:, -1:], (1, 3))

@measure_time
def train_naive_model(X_train, y_train, X_val, y_val):
    """
    Train the naive model and evaluate it on the training and validation sets.
    
    Parameters:
    X_train (ndarray): The training input data.
    y_train (ndarray): The training target data.
    X_val (ndarray): The validation input data.
    y_val (ndarray): The validation target data.
    
    Returns:
    tuple: The RMSE and MAE for the training and validation sets.
    """
    # Generate predictions
    y_train_pred = naive_forecast(X_train)
    y_val_pred = naive_forecast(X_val)

    # Evaluate the predictions
    train_evaluation = evaluate_forecast(y_train, y_train_pred)
    val_evaluation = evaluate_forecast(y_val, y_val_pred)

    return train_evaluation, val_evaluation
    
@measure_time
def test_naive_model(X_test, y_test):
    """
    Test the naive model and evaluate it on the test set.
    
    Parameters:
    X_test (ndarray): The test input data.
    y_test (ndarray): The test target data.
    
    Returns:
    tuple: The RMSE and MAE.
    """
    # Generate predictions
    y_test_pred = naive_forecast(X_test)

    # Evaluate the predictions
    test_evaluation = evaluate_forecast(y_test, y_test_pred)

    return test_evaluation