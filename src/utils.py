# utils.py

from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
import numpy as np
import time


def select_model(models):
    """
    Allow the user to select a model from the available options.
    Repeats until valid input is received.

    Parameters:
    - models (dict): Dictionary of available models {model_name: (train_func, test_func)}

    Returns:
    - tuple: The selected model name and the corresponding train and test functions.
    """
    models_list = list(models.keys())
    models_dict = {str(i + 1): models_list[i] for i in range(len(models_list))}

    while True:
        print("\nAvailable models:")
        for key, value in models_dict.items():
            print(f"{key}: {value}")
        
        choice = input("Select a model by number: ").strip()
        if choice in models_dict:
            model_name = models_dict[choice]
            return model_name, models[model_name]
        print(f"Invalid choice '{choice}'. Please select a valid option.")


def evaluate_forecast(y_true, y_pred, minmax_scaler):
    """
    Evaluates the forecasts using RMSE, MAE, WAPE, and the average RMSE per row.
    
    Parameters:
    y_true (ndarray): The true target values.
    y_pred (ndarray): The predicted values.
    
    Returns:
    dict: RMSE, MAE, MedAE, WAPE, R^2.
    """
    denormalization_factor = 1 / minmax_scaler.scale_
    y_true = y_true * denormalization_factor
    y_pred = y_pred * denormalization_factor

    evaluation = {}

    # RMSE globale
    rmse_global = np.sqrt(mean_squared_error(y_true, y_pred))
    evaluation['rmse'] = rmse_global

    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(y_true, y_pred)
    evaluation['mae'] = mae

    # MedAE (Median Absolute Error)
    medae = median_absolute_error(y_true, y_pred)
    evaluation['medae'] = medae

    # WAPE (Weighted Absolute Percentage Error)
    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    evaluation['wape'] = wape

    # R^2 Score (Coefficient of Determination)
    r2 = r2_score(y_true, y_pred)
    evaluation['r2'] = r2

    return evaluation


def print_model_evaluation(model_name, evaluation):
    print(f'\n{model_name} model evaluation:')
    for test_name, value in evaluation.items():
        print(f' {test_name}: {value:.3f}')


def measure_time(func):
    """
    Decoratore per misurare il tempo di esecuzione di una funzione.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"\nExecution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper



