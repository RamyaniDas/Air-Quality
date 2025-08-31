import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel("Linear_Reg_MV.xlsx")

# Convert all columns to numeric (handle any non-numeric data)
df = df.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values (if any non-numeric values existed)
df.dropna(inplace=True)

# Display first few rows (for debugging)
print("Dataset preview:\n", df.head())

# Insert x0 (intercept term) as the first column
df.insert(0, 'x0', 1)

# Extract features (X) and target (y)
x = df.iloc[:, :-1].values  # All columns except the last one are features
y = df.iloc[:, -1].values   # Last column is the target

# Convert to float for safety
x = x.astype(float)
y = y.astype(float)

# Split data into training (65%) and test (35%) sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0, shuffle=True)

# Compute (X.T * X)
temp1 = np.linalg.inv(x_train.T @ x_train)  # Compute the inverse

# Compute (X.T * y)
temp2 = x_train.T @ y_train

# Compute the weight vector (w)
w = temp1 @ temp2
print("Computed Weights (w):\n", w)

# Function to calculate Mean Squared Error (MSE)
def mse(y_actual, y_predicted):
    return np.mean((y_actual - y_predicted) ** 2)

# Predict values for test set
y_predict = x_test @ w

# Compute MSE for test set (E_out)
print("MSE for test set (E_out):", mse(y_test, y_predict))

# Predict values for training set
y_predict_in = x_train @ w

# Compute MSE for training set (E_in)
print("MSE for training set (E_in):", mse(y_train, y_predict_in))

# Compute baseline MSE (using mean of training y)
baseline_mse = np.mean((y_test - np.mean(y_train)) ** 2)
print("Baseline RMSE:", np.sqrt(baseline_mse))

# Compute RMSE for test set
print("RMSE for test set:", np.sqrt(mse(y_test, y_predict)))

# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_predict, color="blue", label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--", label="Perfect Fit")

# Labels and Title
plt.xlabel("Actual y values")
plt.ylabel("Predicted y values")
plt.title("Actual vs Predicted Values (Test Set)")
plt.legend()
plt.grid(True)
plt.show()
