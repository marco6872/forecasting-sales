# main.py

from data_processing import load_data
from utils import select_model, print_model_evaluation
from models import model_dispatcher



DATASET_CSV_FILENAME = 'weekly_sales_dataset.csv'

TEST_SIZE = 500

REMOVE_OUTLIERS = False
MIN_MAX_NORMALIZATION = True
YEO_JOHNSON_TRANSFORMATION = False


def main():

    # Load and preprocess data
    X_train, y_train, X_test, y_test, minmax_scaler = load_data(
        DATASET_CSV_FILENAME,
        TEST_SIZE,
        REMOVE_OUTLIERS,
        MIN_MAX_NORMALIZATION,
        YEO_JOHNSON_TRANSFORMATION,
    )

    # Allow the user to select a model with validation
    model_choices = model_dispatcher  # Get model choices
    selected_model_name, train_and_test = select_model(model_choices)

    # Train and evaluate the selected model
    print(f"\nTraining and evaluating model: {selected_model_name}")
    
    # Training
    model_evaluation = train_and_test(X_train, y_train, X_test, y_test, minmax_scaler)
    print_model_evaluation(selected_model_name, model_evaluation)

    print()

if __name__ == "__main__":
    main()
