import numpy as np
from typeguard import typechecked

# Add your own imports here

class LeastSquares:
    @typechecked
    def __init__(self) -> None:
        self.w: np.ndarray = np.zeros(1)

    @typechecked
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the linear model to the training data using the least squares method.

        Args:
            X (np.ndarray): The input features for training.
            y (np.ndarray): The target variable for training.
        """

        # self.w = np.linalg.pinv(X.T @ X) @ (X.T @ y)
        self.w = np.linalg.lstsq(X, y, rcond=None)[0]

    @typechecked
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values for new data using the trained linear model.

        Args:
            X (np.ndarray): The input features for prediction.

        Returns:
            np.ndarray: The predicted target values.
        """
        assert(self.w.size > 1)

        return X @ self.w

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

        mse = np.mean((y - y_hat)**2)
        rmse = np.sqrt(mse)

        y_bar = y.mean()
        sse = ((y - y_hat)**2).sum()
        sst = ((y - y_bar)**2).sum()
        r2 = 1 - (sse/sst) if sst != 0 else 0.0

        return rmse, r2
