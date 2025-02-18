import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import evaluate_forecast, measure_time
from models import saved_models_path
from models.deep_learning_utils import create_dataloaders, train_model, test_model


MODEL_FILENAME = 'dl_cnn_model.pth'
NUM_EPOCHS = 400
LEARNING_RATE = 2e-5
BATCH_SIZE = 128


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        # call the parent constructor
        super(ConvolutionalNeuralNetwork, self).__init__()

        # first convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # # second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # first fully connected layer,
        self.fc1 = nn.Linear(in_features=32*9, out_features=64)
        self.dropout1 = nn.Dropout(p=0.2)

        # second fully connected layer, 
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # out = self.pool1(F.relu(self.conv1(x)))
        out = F.relu(self.conv1(x))
        # out = self.pool2(F.relu(self.conv2(out)))
        out = F.relu(self.conv2(out))
        out = torch.flatten(out, 1)
        out = self.dropout1(F.relu(self.fc1(out)))
        out = self.fc2(out)

        return out



@measure_time
def train_and_test_cnn_model(X_train, y_train, X_test, y_test, minmax_scaler):
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
    # Add channel dimension for CNN input
    X_train = X_train[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    # Create data loaders
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, BATCH_SIZE)

    # Initialize the model, criterion, and optimizer
    model = ConvolutionalNeuralNetwork()
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) #, weight_decay=1e-5)

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