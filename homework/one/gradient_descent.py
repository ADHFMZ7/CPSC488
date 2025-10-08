import pandas as pd
import numpy as np
from typeguard import typechecked


class DataLoader:

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_samples = len(X)
        self.indices = np.arange(self.num_samples)
        self.current_index = 0

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= self.num_samples:
            self.current_index = 0
            if self.shuffle:
                np.random.shuffle(self.indices)
            raise StopIteration

        batch_indices = self.indices[self.current_index : self.current_index + self.batch_size]
        batch_data = self.X[batch_indices]
        batch_labels = self.y[batch_indices].reshape((-1, 1))

        self.current_index += self.batch_size
        return batch_data, batch_labels

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class GradientDescent:
    @typechecked
    def __init__(self, learning_rate: float=0.01, n_iterations: int=100, batch_size: int=128) -> None:

        self.weights: np.ndarray = np.zeros(1)

        self.lr = learning_rate
        self.n_iters = n_iterations
        self.batch_size = batch_size


    @typechecked
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the linear model to the training data using mini-batch Gradient Descent.

        Args:
            X (np.ndarray): The input features for training.
            y (np.ndarray): The target variable for training.
        """

        num_features = X.shape[1]

        # Create a dataloader
        loader = DataLoader(X, y, self.batch_size)

        # init model
        self.weights = np.random.normal(0, 0.01, (num_features, 1))

        # Training

        for epoch in range(self.n_iters):

            for batch, (X, y) in enumerate(loader):

                y_hat = X @ self.weights
                residuals = y_hat - y 
                loss = (residuals**2).sum()

                # loss = ((y_hat - y)**2).sum()

                dw = (X.T @ residuals) / X.shape[0] 

                self.weights -= dw * self.lr



    @typechecked
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for new data using the trained linear model.

        Args:
            X (np.ndarray): The input features for prediction.

        Returns:
            np.ndarray: The predicted target values.
        """
        return X @ self.weights


    def eval(self, X: np.ndarray, y: np.ndarray):
        """
        Evaluate the model on given data.

        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).
            y (np.ndarray): True target values of shape (n_samples,).

        Returns:
            tuple:
                rmse (float): Root Mean Squared Error.
                r2 (float): Coefficient of determination.
        """

        y_hat = self.predict(X)

        rsq = (y_hat - y)**2
        rmse = np.sqrt(np.mean(rsq))

        # loss = rsq.sum()

        y_bar = y.mean()
        sse = ((y - y_hat)**2).sum()
        sst = ((y - y_bar)**2).sum()
        r2 = 1 - (sse/sst) if sst != 0 else 0.0

        return rmse, r2


