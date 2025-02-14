import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import dump, load
from utils import evaluate_forecast, measure_time
from models import saved_models_path  # imported from ./models/__init__.py


MODEL_FILENAME = 'linearregression_model.joblib'



@measure_time
def train_and_test_linearregression_model(X_train, y_train, X_test, y_test, minmax_scaler):
    """
    Trains and tests the LinearRegression model.
    
    Parameters:
    X_train (ndarray): Training input features.
    y_train (ndarray): Training output values.
    X_test (ndarray): Testing input features.
    y_test (ndarray): Testing output values.
    minmax_scaler (object): Scaler used for normalizing/denormalizing data.
    
    Returns:
    dict: Evaluation metrics for the forecasts on the test set.
    """
    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Save the trained model
    dump(model, f'{saved_models_path}{MODEL_FILENAME}')

    # Generate predictions on the training data
    y_train_pred = model.predict(X_train)

    # Evaluate the predictions on the training data
    train_evaluation = evaluate_forecast(y_train, y_train_pred, minmax_scaler)

    # Generate predictions on the test data
    y_test_pred = model.predict(X_test)

    # Evaluate the predictions on the test data
    test_evaluation = evaluate_forecast(y_test, y_test_pred, minmax_scaler)

    return test_evaluation

