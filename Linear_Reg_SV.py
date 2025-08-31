import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_excel("Linear_Reg_SV.xlsx", header=None)

# Assign column names
df.columns = ["X", "Y"]

# Identify problematic rows (i.e., non-numeric values)
invalid_x = df[~df["X"].astype(str).str.match(r"^-?\d+(\.\d+)?$")]
invalid_y = df[~df["Y"].astype(str).str.match(r"^-?\d+(\.\d+)?$")]

# If any issues are found, print them and remove problematic rows
if not invalid_x.empty or not invalid_y.empty:
    print("❌ Warning: Non-numeric values detected in dataset!")
    print("Problematic X values:\n", invalid_x)
    print("Problematic Y values:\n", invalid_y)
    df = df.drop(invalid_x.index.union(invalid_y.index))

# Convert to numeric
df["X"] = pd.to_numeric(df["X"], errors='raise')
df["Y"] = pd.to_numeric(df["Y"], errors='raise')

# Scatter plot of dataset
plt.xlim(min(df["X"]) - 1, max(df["X"]) + 1)
plt.ylim(min(df["Y"]) - 5, max(df["Y"]) + 5)
plt.scatter(df["X"], df["Y"], color='red', marker='o')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot Example")
plt.grid(True)
plt.show()

# Split dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(df["X"], df["Y"], test_size=0.2, random_state=42)

# Linear Regression Functions
def mean(values):
    return sum(values) / len(values)

def cov(x, y):
    x_mean = mean(x)
    y_mean = mean(y)
    return sum((i - x_mean) * (j - y_mean) for i, j in zip(x, y)) / len(x)

def var(values):
    values_mean = mean(values)
    return sum((i - values_mean) ** 2 for i in values) / len(values)

# Calculate slope (a) and intercept (b)
a = cov(x_train, y_train) / var(x_train)
b = mean(y_train) - (a * mean(x_train))

print("Slope (a):", a)
print("Intercept (b):", b)

# Plot training and test samples
plt.scatter(x_train, y_train, color="blue", label="Training Sample")
plt.scatter(x_test, y_test, color="green", label="Test Sample")

# Regression Line
x_line = df["X"]  
y_line = a * x_line + b
plt.plot(x_line, y_line, color='red', linestyle='-', label="Regression Line")

# Labels and formatting
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Regression Line with Training & Test Data")
plt.legend()
plt.grid(True)
plt.show()

# Get the predicted values
y_predict = a * x_test + b

# Function to calculate MSE
def mse(y_actual, y_predicted):
    return sum((i - j) ** 2 for i, j in zip(y_actual, y_predicted)) / len(y_actual)

# Compute MSE for out-of-sample data
print("MSE for out-of-sample:", mse(y_test, y_predict))

# Compute MSE for in-sample data
y_predict_train = a * x_train + b
print("MSE for in-sample:", mse(y_train, y_predict_train))

# Compute Baseline RMSE
baseline_mse = np.mean((y_test - np.mean(y_train)) ** 2)
print("Baseline RMSE:", np.sqrt(baseline_mse))

# Compute RMSE for model
print("RMSE:", np.sqrt(mse(y_test, y_predict)))

# Print min and max of Y values
print("Min Y:", df["Y"].min(), "Max Y:", df["Y"].max())
