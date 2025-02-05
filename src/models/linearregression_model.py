import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump, load
from utils import evaluate_forecast, measure_time
from models import saved_models_path  # imported from ./models/__init__.py


MODEL_FILENAME = 'linearregression_model.joblib'


@measure_time
def train_linearregression_model(X_train, y_train, X_val, y_val):
    """
    Train the LinearRegression model using MultiOutputRegressor and evaluate it on the training and validation sets.
    
    Parameters:
    X_train (ndarray): The training input data.
    y_train (ndarray): The training target data.
    X_val (ndarray): The validation input data.
    y_val (ndarray): The validation target data.
    
    Returns:
    tuple: The RMSE and MAE for the training and validation sets.
    """
    # Initialize the base LinearRegression model
    linear_regressor = LinearRegression()

    # Wrap the base regressor with MultiOutputRegressor
    model = MultiOutputRegressor(linear_regressor)

    # Perform GridSearchCV to find the best hyperparameters (if any)
    # Note: LinearRegression has no hyperparameters to tune, so we directly fit the model
    model.fit(X_train, y_train)

    # Save the model
    dump(model, f'{saved_models_path}{MODEL_FILENAME}')

    # Generate predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Evaluate the predictions
    train_evaluation = evaluate_forecast(y_train, y_train_pred)
    val_evaluation = evaluate_forecast(y_val, y_val_pred)

    return train_evaluation, val_evaluation
    

@measure_time
def test_linearregression_model(X_test, y_test):
    """
    Test the LinearRegression model using MultiOutputRegressor and evaluate it on the test set.
    
    Parameters:
    X_test (ndarray): The test input data.
    y_test (ndarray): The test target data.
    
    Returns:
    tuple: The RMSE and MAE for the test set.
    """
    # Load the trained model
    model = load(f'{saved_models_path}{MODEL_FILENAME}')

    # Generate predictions
    y_test_pred = model.predict(X_test)

    # Evaluate the predictions
    test_evaluation = evaluate_forecast(y_test, y_test_pred)

    return test_evaluation
