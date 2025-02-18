# dl_lstm_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import evaluate_forecast, measure_time
from models import saved_models_path
from models.deep_learning_utils import create_dataloaders, train_model, test_model


MODEL_FILENAME = 'dl_lstm_model.pth'
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 128


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=9, hidden_size=50, num_layers=2)
        self.fc = nn.Linear(in_features=50, out_features=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


@measure_time
def train_and_test_lstm_model(X_train, y_train, X_test, y_test, minmax_scaler):
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

    # Create data loaders
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, BATCH_SIZE)

    # Initialize the model, criterion, and optimizer
    model = LSTM()
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, test_loader, NUM_EPOCHS)

    # Save the model
    torch.save(model.state_dict(), f'{saved_models_path}{MODEL_FILENAME}')
    print(f'\nModel saved to {saved_models_path}{MODEL_FILENAME}')

    # Test the model
    y_test_pred = test_model(model, test_loader)

    # Evaluate the predictions
    test_evaluation = evaluate_forecast(y_test, y_test_pred.squeeze(), minmax_scaler)

    return test_evaluation