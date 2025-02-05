import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump, load
from utils import evaluate_forecast, measure_time
from models import saved_models_path  # imported from ./models/__init__.py
from sklearn.model_selection import GridSearchCV, cross_val_score


MODEL_FILENAME = 'randomforest_model.joblib'


@measure_time
def train_randomforest_model(X_train, y_train, X_val, y_val):
    """
    Train the RandomForest model using MultiOutputRegressor and evaluate it on the training and validation sets.
    
    Parameters:
    X_train (ndarray): The training input data.
    y_train (ndarray): The training target data.
    X_val (ndarray): The validation input data.
    y_val (ndarray): The validation target data.
    
    Returns:
    tuple: The RMSE and MAE for the training and validation sets.
    """
    # Initialize the base RandomForest regressor
    randomforest_regressor = RandomForestRegressor(random_state=42)

    # Define the parameter grid for GridSearchCV
    # param_grid = {
    #     'estimator__n_estimators': [100, 200],
    #     'estimator__max_depth': [None, 5, 10, 20, 30],
    #     'estimator__min_samples_split': [2, 5, 10],
    #     'estimator__min_samples_leaf': [1, 2, 4],
    #     'estimator__max_features': ['auto', 'sqrt', 'log2'],
    #     'estimator__bootstrap': [True, False]
    # }

    # semplificato per test veloci
    param_grid = {
        'estimator__n_estimators': [100],
        'estimator__max_depth': [5],
        'estimator__min_samples_split': [5],
        'estimator__min_samples_leaf': [2],
        'estimator__bootstrap': [True, False]
    }

    # Wrap the base regressor with MultiOutputRegressor
    model = MultiOutputRegressor(randomforest_regressor)

    # Perform GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error', verbose=3)
    grid_search.fit(X_train, y_train)

    # Best model from GridSearchCV
    best_model = grid_search.best_estimator_
    print_best_params(grid_search.best_params_)

    # Save the best model
    dump(best_model, f'{saved_models_path}{MODEL_FILENAME}')

    # Generate predictions
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)

    # Evaluate the predictions
    train_evaluation = evaluate_forecast(y_train, y_train_pred)
    val_evaluation = evaluate_forecast(y_val, y_val_pred)

    return train_evaluation, val_evaluation


@measure_time
def test_randomforest_model(X_test, y_test):
    """
    Test the RandomForest model using MultiOutputRegressor and evaluate it on the test set.
    
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


def print_best_params(best_params):
    """
    Print the best parameters found by GridSearchCV.
    
    Parameters:
    best_params (dict): The best parameters found by GridSearchCV.
    """
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

