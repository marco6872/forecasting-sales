import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset



def create_dataloaders(X_train, y_train, X_test, y_test, batch_size):
    """
    Create data loaders for training and testing datasets.
    
    Parameters:
    X_train (np.array): Training features.
    y_train (np.array): Training target values.
    X_test (np.array): Test features.
    y_test (np.array): Test target values.
    batch_size (int): Batch size for data loading.
    
    Returns:
    tuple: Training and testing data loaders.
    """
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                  torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), 
                                 torch.tensor(y_test, dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model, criterion, optimizer, train_loader, num_epochs):
    """
    Train the given model.
    
    Parameters:
    model (nn.Module): The neural network model to be trained.
    criterion (nn.Module): The loss function.
    optimizer (optim.Optimizer): The optimizer for training.
    train_loader (DataLoader): DataLoader for the training set.
    num_epochs (int): The number of training epochs.
    
    Returns:
    np.array: Predictions on the training set.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    print('Training complete')

    predictions = test_model(model, train_loader)
    return predictions


def test_model(model, test_loader):
    """
    Test the given model.
    
    Parameters:
    model (nn.Module): The neural network model to be tested.
    test_loader (DataLoader): DataLoader for the test set.
    
    Returns:
    np.array: Predictions on the test set.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            predictions.append(outputs.numpy())
    predictions = np.vstack(predictions)
    return predictions