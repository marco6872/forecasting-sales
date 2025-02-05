# main.py

import os
from data_processing import load_data
from utils import ask_user, select_model, print_model_evaluation
from models import model_dispatcher   # imported from ./models/__init__.py

def main():
    """
    Main function to act as a user interface for the project.
    """
    # Ask the user for input with validation
    preprocess = ask_user("\nDo you want to preprocess the data? (y/n): ")
    show_plots = False
    if preprocess:
        show_plots = ask_user("Do you want to show plots during preprocessing? (y/n): ")

    # Load or preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(preprocess, show_plots)

    # Allow the user to select a model with validation
    model_choices = model_dispatcher  # Get model choices
    selected_model_name, train_func, test_func = select_model(model_choices)

    # Train and evaluate the selected model
    print(f"\nTraining and evaluating model: {selected_model_name}")
    
    # # Training
    # train_rmse, train_mae, val_rmse, val_mae = train_func(X_train, y_train, X_val, y_val)
    # print('\nTrain data evaluation:')
    # print(f' RMSE: {train_rmse:.2f}\n MAE: {train_mae:.2f}')
    # print('Validation data evaluation:')
    # print(f' RMSE: {val_rmse:.2f}\n MAE: {val_mae:.2f}')

    # # Testing
    # test_rmse, test_mae = test_func(X_test, y_test)
    # print('\nTest data evaluation:')
    # print(f' RMSE: {test_rmse:.2f}\n MAE: {test_mae:.2f}')


    train_evaluation, val_evaluation = train_func(X_train, y_train, X_val, y_val)
    print_model_evaluation("Train", train_evaluation)
    print_model_evaluation("Validation", val_evaluation)

    test_evaluation = test_func(X_test, y_test)
    print_model_evaluation("Test", test_evaluation)

    print()

if __name__ == "__main__":
    main()
