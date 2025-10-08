import pandas as pd
import numpy as np
from itertools import combinations_with_replacement
# from sklearn.preprocessing import PolynomialFeatures
from evaluate_models import add_polynomial_features
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import time

from gradient_descent import GradientDescent

# Load data
data = pd.read_csv('../../data/GasProperties.csv')
y = data['Idx'].to_numpy().reshape(-1, 1)
X = data.drop('Idx', axis=1).to_numpy()

# Standardize
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# print(f"{'Order':<6} | {'Train RMSE':<12} | {'Train R2':<10} | {'Train Time':<12} | {'Test RMSE':<12} | {'Test R2':<10} |")
X_train = add_polynomial_features(X_train, 1)

model = GradientDescent()

model.fit(X_train, y_train)


# for order in range(1, 6):
#     # Polynomial features
#     poly = PolynomialFeatures(degree=order, include_bias=True)
#     X_train_poly = poly.fit_transform(X_train)
#     X_test_poly = poly.transform(X_test)
#
#     # Fit Linear Regression
#     model = LinearRegression()
#
#     start = time.time()
#     model.fit(X_train_poly, y_train)
#     end = time.time()
#     train_time = end - start
#
#     # Predictions
#     y_train_pred = model.predict(X_train_poly)
#     y_test_pred = model.predict(X_test_poly)
#
#     # Metrics
#     train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#     test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#     train_r2 = r2_score(y_train, y_train_pred)
#     test_r2 = r2_score(y_test, y_test_pred)
#
#     print(f"{order:<6} | {train_rmse:<12.6e} | {train_r2:<10.6e} | {train_time:<12.4f} | {test_rmse:<12.6e} | {test_r2:<10.6e} |")

