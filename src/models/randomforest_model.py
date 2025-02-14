import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump, load
from utils import evaluate_forecast, measure_time
from models import saved_models_path  # imported from ./models/__init__.py
from sklearn.model_selection import GridSearchCV, cross_val_score

# Define the filename for saving the trained model
MODEL_FILENAME = 'randomforest_model.joblib'

def print_best_params(best_params):
    """
    Print the best parameters found by GridSearchCV.
    
    Parameters:
    best_params (dict): The best parameters found by GridSearchCV.
    """
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

@measure_time
def train_and_test_randomforest_model(X_train, y_train, X_test, y_test, minmax_scaler):
    """
    Train and test a RandomForestRegressor model using GridSearchCV.
    
    Parameters:
    X_train (np.array): Training features.
    y_train (np.array): Training target values.
    X_test (np.array): Test features.
    y_test (np.array): Test target values.
    minmax_scaler (object): Scaler object for inverse transforming the predictions.
    
    Returns:
    test_evaluation (dict): Evaluation metrics for the test set predictions.
    """

    # Define the parameter grid for GridSearchCV
    # param_grid = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [None, 5, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['auto', 'sqrt', 'log2'],
    #     'bootstrap': [True, False]
    # }
    param_grid = {
        'n_estimators': [100],
        'max_depth': [5],
        'min_samples_split': [5],
        'min_samples_leaf': [2],
        'bootstrap': [True]
    }

    # Initialize the RandomForestRegressor model
    model = RandomForestRegressor(random_state=42)

    # Perform GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error', verbose=3)
    grid_search.fit(X_train, y_train)

    # Best model from GridSearchCV
    best_model = grid_search.best_estimator_
    print_best_params(grid_search.best_params_)

    # Save the best model
    dump(best_model, f'{saved_models_path}{MODEL_FILENAME}')

    # Generate predictions for the training set
    y_train_pred = best_model.predict(X_train)

    # Evaluate the predictions for the training set
    train_evaluation = evaluate_forecast(y_train, y_train_pred, minmax_scaler)

    # Generate predictions for the test set
    y_test_pred = best_model.predict(X_test)

    # Evaluate the predictions for the test set
    test_evaluation = evaluate_forecast(y_test, y_test_pred, minmax_scaler)

    return test_evaluation
