import numpy as np
import torch as t
import time
from mlp import MLP, train_model

def evaluate_model(model, X_train, y_train, X_test, y_test):
    
    if isinstance(model, MLP):

        device = next(model.parameters()).device

        X_train, y_train = t.tensor(X_train).to(device), t.tensor(y_train).to(device)
        X_test , y_test = t.tensor(X_test).to(device), t.tensor(y_test).to(device)

        y_hat = model.forward(X_train)
        
        train_rmse = t.sqrt(t.mean((y_hat - y_train)**2))
        train_r2 = 1 - (t.sum((y_hat - y_train)**2) / t.sum((y_train - y_train.mean())**2))

        y_hat = model.forward(X_test)

        test_rmse = t.sqrt(t.mean((y_hat - y_test)**2))
        test_r2 = 1 - (t.sum((y_hat - y_test)**2) / t.sum((y_test - y_test.mean())**2))

    
    else:
        train_rmse, train_r2 = model.eval(X_train, y_train)
        test_rmse, test_r2 = model.eval(X_test, y_test)

    return train_rmse, train_r2, test_rmse, test_r2

def create_loader(X, y, batch_size=64, shuffle=True):

    dataset = t.utils.data.TensorDataset(t.tensor(X), t.tensor(y))
    loader = t.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return loader
