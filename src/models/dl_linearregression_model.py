import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils import evaluate_forecast, measure_time
from models import saved_models_path  # imported from ./models/__init__.py
from models.deep_learning_utils import create_dataloaders, train_model, test_model


MODEL_FILENAME = 'dl_linearregression_model.pth'
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
BATCH_SIZE = 128


# Define different neural network classes

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the linear regression model.
        
        Parameters:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        """
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for the linear regression model.
        
        Parameters:
        x (torch.Tensor): Input tensor.
        
        Returns:
        torch.Tensor: Output tensor.
        """
        return self.linear(x)


@measure_time
def train_and_test_torch_linearregression_model(X_train, y_train, X_test, y_test, minmax_scaler):
    """
    Train and test the Linear Regression model class.
    
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
    model = LinearRegressionModel(input_dim, output_dim)
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
