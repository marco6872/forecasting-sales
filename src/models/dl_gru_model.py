# dl_gru_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchinfo import summary

from utils import evaluate_forecast, measure_time
from models import saved_models_path
from models.deep_learning_utils import create_dataloaders, train_model, test_model, set_seed, get_device

from visualize import plot_real_vs_predicted


MODEL_FILENAME = 'dl_gru_model.pth'
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 128


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        self.lstm = nn.GRU(input_size=9, hidden_size=32)
        self.fc = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


@measure_time
def train_and_test_gru_model(X_train, y_train, X_test, y_test, minmax_scaler):
    """
    Train and test the Fully Connected Network model class.
    
    Parameters:
    X_train (np.array): Training features.
    y_train (np.array): Training target values.
    X_test (np.array): Test features.
    y_test (np.array): Test target values.
    minmax_scaler (object): Scaler object for inverse transforming the predictions.
    
    Returns:
    dict: Evaluation metrics for the test set predictions.
    """

    # Set a seed for reproducibility
    set_seed(42)

    # Get device
    device = get_device()
    print(f'Using device: {device}')

    # Create data loaders
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, BATCH_SIZE, device)

    # Initialize the model, criterion, and optimizer
    model = GRU().to(device)
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # train parameters
    print(f'\nLearning Rate: {LEARNING_RATE}')
    print(f'Epochs: {NUM_EPOCHS}')
    print(f'Batch Size: {BATCH_SIZE}')

    # model summary
    summary(model, input_size=(BATCH_SIZE, 9))

    # Train the model
    train_model(model, criterion, optimizer, train_loader, test_loader, NUM_EPOCHS, device)

    # Save the model
    torch.save(model.state_dict(), f'{saved_models_path}{MODEL_FILENAME}')
    print(f'\nModel saved to {saved_models_path}{MODEL_FILENAME}')

    # Test the model
    y_test_pred = test_model(model, test_loader, device)

    # Evaluate the predictions
    test_evaluation = evaluate_forecast(y_test, y_test_pred.squeeze(), minmax_scaler)

    plot_real_vs_predicted(y_test, y_test_pred.squeeze())

    return test_evaluation
