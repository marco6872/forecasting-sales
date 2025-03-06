# ddl_fcn_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torchinfo import summary

from utils import evaluate_forecast, measure_time
from models import saved_models_path
from models.deep_learning_utils import create_dataloaders, train_model, test_model, set_seed

from visualize import plot_real_vs_predicted


MODEL_FILENAME = 'dl_fcn_model.pth'
NUM_EPOCHS = 3000
LEARNING_RATE = 2e-5
BATCH_SIZE = 128


class FullyConnectedNetwork(nn.Module):
    def __init__(self):
        """
        Initialize the fully connected network.
        
        Parameters:
        input_size (int): The number of input features.
        hidden_size (int): The number of units in the hidden layers.
        output_size (int): The number of output features.
        """
        super(FullyConnectedNetwork, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.relu1 = nn.LeakyReLU(0.1)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)
    
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
        return out


# class FullyConnectedNetwork(nn.Module):
#     def __init__(self):
#         """
#         Initialize the fully connected network.
        
#         Parameters:
#         input_size (int): The number of input features.
#         hidden_size (int): The number of units in the hidden layers.
#         output_size (int): The number of output features.
#         """
#         super(FullyConnectedNetwork, self).__init__()
#         self.fc1 = nn.Linear(9, 64)
#         self.relu1 = nn.ReLU()
#         self.drop1 = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(64, 128)
#         self.relu2 = nn.ReLU()
#         self.drop2 = nn.Dropout(0.3)
#         self.fc3 = nn.Linear(128, 64)
#         self.relu3 = nn.ReLU()
#         self.drop3 = nn.Dropout(0.2)
#         self.fc4 = nn.Linear(64, 1)
    
#     def forward(self, x):
#         """
#         Forward pass for the fully connected network.
        
#         Parameters:
#         x (torch.Tensor): Input tensor.
        
#         Returns:
#         torch.Tensor: Output tensor.
#         """
#         out = self.fc1(x)
#         out = self.relu1(out)
#         out = self.drop1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.drop2(out)
#         out = self.fc3(out)
#         out = self.relu3(out)
#         out = self.drop3(out)
#         out = self.fc4(out)
#         return out



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

    # Set a seed for reproducibility
    set_seed(42)

    input_dim = X_train.shape[1]
    output_dim = 1

    # Create data loaders
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, BATCH_SIZE)

    # Initialize the model, criterion, and optimizer
    model = FullyConnectedNetwork()
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train parameters
    print(f'\nLearning Rate: {LEARNING_RATE}')
    print(f'Epochs: {NUM_EPOCHS}')
    print(f'Batch Size: {BATCH_SIZE}')

    # model summary
    summary(model, input_size=(BATCH_SIZE, 9))

    # Train the model
    train_model(model, criterion, optimizer, train_loader, test_loader, NUM_EPOCHS)

    # Save the model
    torch.save(model.state_dict(), f'{saved_models_path}{MODEL_FILENAME}')
    print(f'\nModel saved to {saved_models_path}{MODEL_FILENAME}')

    # Test the model
    y_test_pred = test_model(model, test_loader)

    # Evaluate the predictions
    test_evaluation = evaluate_forecast(y_test, y_test_pred.squeeze(), minmax_scaler)

    plot_real_vs_predicted(y_test, y_test_pred.squeeze())

    return test_evaluation
