# dl_cnn_model.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary

from utils import evaluate_forecast, measure_time
from models import saved_models_path
from models.deep_learning_utils import create_dataloaders, train_model, test_model, set_seed

from visualize import plot_real_vs_predicted

# Define model parameters
MODEL_FILENAME = 'dl_cnn_model.pth'
NUM_EPOCHS = 2000
LEARNING_RATE = 1e-5
BATCH_SIZE = 128


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        # Call the parent constructor
        super(ConvolutionalNeuralNetwork, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        # self.relu1 = nn.LeakyReLU(0.1)
        # self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        # self.relu2 = nn.LeakyReLU(0.1)
        # self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16*9, out_features=128)
        self.relufc1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=128, out_features=1)

    def forward(self, x):
        # Pass the input through the first convolutional layer
        out = self.conv1(x)
        # out = self.relu1(self.conv1(x))
        
        # Pass the result through the second convolutional layer and apply LeakyReLU activation
        # out = self.relu2(self.conv2(out))
        out = self.conv2(out)
        
        # Flatten the output for the fully connected layers
        out = torch.flatten(out, 1)
        
        # Pass through the first fully connected layer with dropout and ReLU activation
        out = self.dropout1(self.relufc1(self.fc1(out)))
        # out = self.dropout1(self.fc1(out))
        
        # Pass through the second fully connected layer to get the final output
        out = self.fc2(out)

        return out

@measure_time
def train_and_test_cnn_model(X_train, y_train, X_test, y_test, minmax_scaler):
    """
    Train and test the Convolutional Neural Network model class.
    
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

    # Set a seed for reproducibility
    set_seed(42)

    # Create data loaders
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, BATCH_SIZE)

    # Initialize the model, criterion, and optimizer
    model = ConvolutionalNeuralNetwork()
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # train parameters
    print(f'\nLearning Rate: {LEARNING_RATE}')
    print(f'Epochs: {NUM_EPOCHS}')
    print(f'Batch Size: {BATCH_SIZE}')

    # model summary
    summary(model, input_size=(BATCH_SIZE, 1, 9))

    # Train the model
    train_model(model, criterion, optimizer, train_loader, test_loader, NUM_EPOCHS)

    # Save the trained model to disk
    torch.save(model.state_dict(), f'{saved_models_path}{MODEL_FILENAME}')
    print(f'\nModel saved to {saved_models_path}{MODEL_FILENAME}')

    # Test the model on the test set
    y_test_pred = test_model(model, test_loader)

    # Evaluate the predictions against the actual values
    test_evaluation = evaluate_forecast(y_test, y_test_pred.squeeze(), minmax_scaler)

    plot_real_vs_predicted(y_test, y_test_pred.squeeze())

    return test_evaluation


