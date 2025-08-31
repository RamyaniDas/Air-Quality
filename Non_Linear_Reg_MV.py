import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Load the dataset and ensure correct column names
df = pd.read_excel("Non_Linear_Regression_MV.xlsx", header=None)

# Rename columns manually (assuming it has 3 columns)
df.columns = ['x1', 'x2', 'y']

# Ensure there are no extra spaces in column names
df.columns = df.columns.str.strip()

# Display dataset preview
print("Dataset preview:")
print(df.head())

# Create polynomial features
df.insert(0, 'x1x2^2', df['x1'] * (df['x2'] ** 2))
df.insert(0, 'x1^2x2', (df['x1'] ** 2) * df['x2'])
df.insert(0, 'x1^2x2^2', (df['x1'] ** 2) * (df['x2'] ** 2))
df.insert(5, 'x0', 1)  # Bias term

# Prepare feature matrix X and target vector y
X = df.iloc[:, [0, 1, 2, 3, 4, 5]].values
y = df.iloc[:, 6].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)

# Compute weights using Normal Equation
temp1 = np.linalg.inv(X_train.T @ X_train)  # Compute (X^T * X)^-1
temp2 = X_train.T @ y_train  # Compute X^T * y
w = temp1 @ temp2  # Compute w

print("Computed Weights:")
print(w)

# Define MSE function
def mse(y_actual, y_pred):
    return np.mean((y_actual - y_pred) ** 2)

# Compute MSE for test data
y_predict = X_test @ w
print("MSE for out sample: ", mse(y_test, y_predict))

# Compute MSE for training data
y_predict_in = X_train @ w
print("MSE for in sample: ", mse(y_train, y_predict_in))

# Compute baseline RMSE
baseline_mse = np.mean((y_test - np.mean(y_train)) ** 2)
print("Baseline RMSE:", np.sqrt(baseline_mse))
print("RMSE:", np.sqrt(mse(y_test, y_predict)))

# Plot Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_predict, color="blue", label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual y values")
plt.ylabel("Predicted y values")
plt.title("Actual vs Predicted Values (Test Set)")
plt.legend()
plt.grid(True)
plt.show()

# ---- Second Model with Higher-Order Polynomial Features ---- #
df = pd.read_excel("Non_Linear_Regression_MV.xlsx", header=None)
df.columns = ['x1', 'x2', 'y']
df.columns = df.columns.str.strip()  # Remove any unwanted spaces

# Add higher-order polynomial features
df.insert(0, 'x1x2', df['x1'] * df['x2'])
df.insert(0, 'x2^2', df['x2'] ** 2)
df.insert(0, 'x1^2', df['x1'] ** 2)
df.insert(0, 'x2^3', df['x2'] ** 3)
df.insert(0, 'x1^3', df['x1'] ** 3)
df.insert(7, 'x0', 1)  # Bias term

# Prepare features and target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)

# Compute weights using Normal Equation
temp1 = np.linalg.inv(X_train.T @ X_train)
temp2 = X_train.T @ y_train
w = temp1 @ temp2

print("Computed Weights for second model:")
print(w)

# Compute MSE
y_predict = X_test @ w
print("MSE for out sample: ", mse(y_test, y_predict))

y_predict_train = X_train @ w
print("MSE for in sample: ", mse(y_train, y_predict_train))

baseline_mse = np.mean((y_test - np.mean(y_train)) ** 2)
print("Baseline RMSE:", np.sqrt(baseline_mse))
print("RMSE:", np.sqrt(mse(y_test, y_predict)))

# Plot Actual vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_predict, color="blue", label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual y values")
plt.ylabel("Predicted y values")
plt.title("Actual vs Predicted Values (Test Set)")
plt.legend()
plt.grid(True)
plt.show()

# ---- Ridge Regression with Polynomial Features ---- #
# Split dataset for Ridge Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Create cubic polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Apply Ridge Regression (L2 Regularization)
ridge = Ridge(alpha=0.1)  # Adjust alpha for regularization strength
ridge.fit(X_train_poly, y_train)

# Predictions
y_train_pred = ridge.predict(X_train_poly)
y_test_pred = ridge.predict(X_test_poly)

# Compute MSE
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("MSE (Train):", mse_train)
print("MSE (Test):", mse_test)

# Plot Actual vs Predicted values for Ridge Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color="blue", label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")
plt.xlabel("Actual y values")
plt.ylabel("Predicted y values")
plt.title("Actual vs Predicted Values (Test Set) - Ridge Regression")
plt.legend()
plt.grid(True)
plt.show()
