import pandas as pd
import numpy as np
import time
import torch as t
from typeguard import typechecked
from itertools import combinations_with_replacement

from least_square import LeastSquares
from gradient_descent import GradientDescent
from mlp import setup_model, train_model
from utils import evaluate_model, create_loader

@typechecked
def preprocess(X: np.ndarray, y: np.ndarray, to_tensor: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesses the input data by splitting it into training and testing sets
    and scaling the features. Optionally converts data to PyTorch tensors.

    Args:
        X (np.ndarray): The input features as a NumPy array.
        y (np.ndarray): The target variable as a NumPy array.
        to_tensor (bool): If True, convert the processed data to PyTorch tensors.

    Returns:
        tuple: A tuple containing the processed data. The exact contents depend on `to_tensor`:
            - If to_tensor is False: (X_train_scaled, X_test_scaled, y_train, y_test)
            - If to_tensor is True: (X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor)
    """
   
    # X = (X - X.mean(axis=0))/X.std(axis=0)

    X = (X - X.min()) / (X.max() - X.min())

    split = int(len(X) * 0.8)
    train_X, train_y = X[:split], y[:split]
    test_X, test_y =   X[split:], y[split:]

    return train_X, test_X, train_y, test_y


@typechecked
def add_polynomial_features(x: np.ndarray, order: int) -> np.ndarray:
    """
    Adds polynomial features to the input array X up to the specified order.

    Args:
        x (np.ndarray): The input numpy array of features. Each column represents a feature.
        order (int): The maximum degree of the polynomial features to add.

    Returns:
        np.ndarray: A new numpy array with the original features and the added
                    polynomial features
    """

    n_samples, n_features = x.shape
    features = [np.ones((n_samples, 1))]  

    for deg in range(1, order + 1):
        for comb in combinations_with_replacement(range(n_features), deg):
            new_feature = np.prod(x[:, comb], axis=1, keepdims=True)
            features.append(new_feature)

    return np.hstack(features)


def run_tests() -> None:
    """
    Executes a comprehensive evaluation of Least Squares, Gradient Descent, and MLP models
    on the 'GasProperties.csv' dataset. It assesses model performance across different
    polynomial degrees for feature engineering.

    The evaluation process includes:
    - Data loading and optional subsampling.
    - Iterating through polynomial degrees (1 to 5) to observe their impact on model performance.
    - Training and evaluating Least Squares, Gradient Descent, and MLP models.
    - Calculating Root Mean Squared Error (RMSE) and R-squared (R^2) for both training
      and testing datasets.
    - Recording and displaying training times for each model.
    """
    data = pd.read_csv('./data/GasProperties.csv')
    y = data.Idx.to_numpy().reshape(-1, 1)

    X = data.drop('Idx', axis=1).to_numpy()

    X_train_scaled, X_test_scaled, y_train, y_test = preprocess(X, y, to_tensor=False)

    print(X_train_scaled.shape)

    assert(X_train_scaled.shape[0] == y_train.shape[0])
    assert(X_test_scaled.shape[0] == y_test.shape[0])

    print("\n\nLeast Squares Metrics")
    print(f"{'Order':<6} | {'Train RMSE':<12} | {'Train R2':<10} | {'Train Time':<12} | {'Test RMSE':<12} | {'Test R2':<10} |")

    for order in range(1, 6):
        X_in = add_polynomial_features(X_train_scaled, order)
        # print(order, X_in.shape)
        model = LeastSquares()

        start = time.time()
        model.fit(X_in, y_train)
        end = time.time()
        train_time = end - start

        test_in = add_polynomial_features(X_test_scaled, order)

        train_rmse, train_r2, test_rmse, test_r2 = evaluate_model(model, X_in, y_train, test_in, y_test)

        print(f"{order:<6} | {train_rmse:<12.6e} | {train_r2:<10.6e} | {train_time:<12.4f} | {test_rmse:<12.6e} | {test_r2:<10.6e} |")


    # print("\n\nGradient Descent Metrics")
    # print(f"{'Order':<6} | {'Train RMSE':<12} | {'Train R2':<10} | {'Train Time':<12} | {'Test RMSE':<12} | {'Test R2':<10} |")
    # for order in range(1, 6):
    #     X_in = add_polynomial_features(X_train_scaled, order)
    #     model = GradientDescent(n_iterations=1000, batch_size=128)
    #
    #     start = time.time()
    #     model.fit(X_in, y_train)
    #     end = time.time()
    #     train_time = end - start
    #
    #     test_in = add_polynomial_features(X_test_scaled, order)
    #
    #     train_rmse, train_r2, test_rmse, test_r2 = evaluate_model(model, X_in, y_train, test_in, y_test)
    #
    #     print(f"{order:<6} | {train_rmse:<12.6e} | {train_r2:<10.6e} | {train_time:<12.4f} | {test_rmse:<12.6e} | {test_r2:<10.6e} |")
    #
    # device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    # print(f'Device available: {device}')
    #
    # print("\n\nMLP Metrics")
    # print(f"{'Order':<6} | {'Train RMSE':<12} | {'Train R2':<10} | {'Train Time':<12} | {'Test RMSE':<12} | {'Test R2':<10} |")
    #
    # for order in range(1, 6):
    #     X_in = add_polynomial_features(X_train_scaled, order)
    #
    #     model, criterion, optim = setup_model(X_in.shape[1], [10, 10], 1, device=device)
    #
    #     start = time.time()
    #     train_model(model, create_loader(X_in, y_train, batch_size=128), criterion, optim, num_epochs=10)
    #     end = time.time()
    #     train_time = end - start
    #
    #     test_in = add_polynomial_features(X_test_scaled, order)
    #
    #     train_rmse, train_r2, test_rmse, test_r2 = evaluate_model(model, X_in, y_train, test_in, y_test)
    #
    #     print(f"{order:<6} | {train_rmse:<12.6e} | {train_r2:<10.6e} | {train_time:<12.4f} | {test_rmse:<12.6e} | {test_r2:<10.6e} |")



if __name__ == "__main__":
    run_tests()
