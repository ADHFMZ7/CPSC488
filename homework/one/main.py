import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from least_square import LeastSquares
from gradient_descent import GradientDescent
from mlp import MLP, setup_model, train_model
from evaluate_models import add_polynomial_features, preprocess

seed = 42

data = pd.read_csv('../../data/GasProperties.csv')
data_shuffled = data.sample(frac=1, random_state=seed)
data_shuffled = data_shuffled.reset_index(drop=True)

X = np.hstack(
    tuple(data_shuffled[col_name].to_numpy().reshape((-1, 1)) for col_name in data_shuffled.columns if col_name != 'Idx')
)
Y = data_shuffled.Idx.to_numpy().reshape((-1, 1))

X_train, X_test, Y_train, Y_test = preprocess(X, Y)

X_train = add_polynomial_features(X_train, 3)

model = GradientDescent(n_iterations=10000)

model.fit(X_train, Y_train)

train_rmse, train_r2 = model.eval(X_train, Y_train)
test_rmse, test_r2 = model.eval(X_test, Y_test)

print(train_rmse, train_r2)
print(test_rmse, test_r2)










# X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
# Y_train_tensor = torch.tensor(Y_train, dtype=torch.float64)
# train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
#
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
#
#
# X_test = np.hstack(
#     tuple(test[col_name].to_numpy().reshape((-1, 1)) for col_name in test.columns if col_name != 'Idx')
# )
# Y_test = test.Idx.to_numpy().reshape((-1, 1))
#
# X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
# Y_test_tensor = torch.tensor(Y_test, dtype=torch.float64)
#
# test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
#
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
#
# model, criterion, optim = setup_model(X_train.shape[1], [10, 100, 10], 1, torch.device('cpu'))
# model = train_model(model, train_loader, criterion, optim, 10)
#
# with torch.no_grad():
#     y_hat = model(X_test_tensor).numpy()
#     y = Y_test_tensor.numpy()
#
#     mse = np.mean((y - y_hat)**2)
#     rmse = np.sqrt(mse)
#
#     y_bar = y.mean()
#     sse = ((y - y_hat)**2).sum()
#     sst = ((y - y_bar)**2).sum()
#     r2 = 1 - (sse/sst) if sst != 0 else 0.0
#
#     print(rmse, r2)
