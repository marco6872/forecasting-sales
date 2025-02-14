import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import evaluate_forecast, measure_time
from models import saved_models_path  # imported from ./models/__init__.py
from models.deep_learning_utils import create_dataloaders, train_model, test_model


MODEL_FILENAME = 'dl_fcn_model.pth'
HIDDEN_SIZE = 64
NUM_EPOCHS = 2000
LEARNING_RATE = 2e-5
BATCH_SIZE = 128


class FullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the fully connected network.
        
        Parameters:
        input_size (int): The number of input features.
        hidden_size (int): The number of units in the hidden layers.
        output_size (int): The number of output features.
        """
        super(FullyConnectedNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass for the fully connected network.
        
        Parameters:
        x (torch.Tensor): Input tensor.
        
        Returns:
        torch.Tensor: Output tensor.
        """
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        return out


@measure_time
def train_and_test_fcn_model(X_train, y_train, X_test, y_test, minmax_scaler):
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
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    # Create data loaders
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, BATCH_SIZE)

    # Initialize the model, criterion, and optimizer
    model = FullyConnectedNetwork(input_dim, HIDDEN_SIZE, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_model(model, criterion, optimizer, train_loader, NUM_EPOCHS)

    # Save the model
    torch.save(model.state_dict(), f'{saved_models_path}{MODEL_FILENAME}')
    print(f'Model saved to {saved_models_path}{MODEL_FILENAME}')

    # Test the model
    y_test_pred = test_model(model, test_loader)

    # Evaluate the predictions
    test_evaluation = evaluate_forecast(y_test, y_test_pred.squeeze(), minmax_scaler)

    return test_evaluation
