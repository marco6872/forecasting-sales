# utils.py

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import time


def ask_user(prompt):
    """
    Ask the user a yes/no question and return a boolean.
    Repeats until valid input is received.

    Parameters:
    - prompt (str): The question to ask the user.

    Returns:
    - bool: True if the user answers 'y', False if 'n'.
    """
    while True:
        response = input(prompt).strip().lower()
        if response in ('y', 'yes'):
            return True
        elif response in ('n', 'no'):
            return False
        print("Invalid input. Please answer with 'y'/'yes' or 'n'/'no'.")

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
            return model_name, models[model_name][0], models[model_name][1]
        print(f"Invalid choice '{choice}'. Please select a valid option.")



def evaluate_forecast(y_true, y_pred):
    """
    Evaluates the forecasts using RMSE, MAE, and WAPE.
    
    Parameters:
    y_true (ndarray): The true target values.
    y_pred (ndarray): The predicted values.
    
    Returns:
    tuple: The RMSE, MAE, and WAPE.
    """
    evaluation = {}

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    evaluation['rmse'] = rmse

    mae = mean_absolute_error(y_true, y_pred)
    evaluation['mae'] = mae

    wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100  # WAPE calcolato in percentuale
    evaluation['wape'] = wape

    return evaluation


def print_model_evaluation(dataset_name, evaluation):
    print(f'\n{dataset_name} data evaluation:')
    for test_name, value in evaluation.items():
        print(f' {test_name}: {value:.2f}')


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


