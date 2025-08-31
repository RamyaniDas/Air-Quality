import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_excel("Non_Linear_Reg_SV.xlsx")

# Ensure x and y are numeric
x = pd.to_numeric(df.iloc[:, 0], errors='coerce')  # Convert first column to numbers
y = pd.to_numeric(df.iloc[:, 1], errors='coerce')  # Convert second column to numbers

# Drop any NaN values if conversion fails for some rows
valid_indices = ~(np.isnan(x) | np.isnan(y))
x = x[valid_indices]
y = y[valid_indices]

# Convert x to a numpy array for further calculations
x = np.array(x)
y = np.array(y)

# Plot the original data points
plt.scatter(x, y, color="blue")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Non_Linear_Reg_SV")
plt.grid(True)
plt.xlim(min(x) - 1, max(x) + 1)  # Expands X-axis range
plt.ylim(min(y) - 5, max(y) + 5)  # Expands Y-axis range
plt.show()

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Computing required summations for quadratic regression
n = len(x_train)
mean_x = np.mean(x_train)
mean_x2 = np.mean(x_train**2)
mean_x3 = np.mean(x_train**3)
mean_x4 = np.mean(x_train**4)
mean_y = np.mean(y_train)
mean_yx = np.mean(y_train * x_train)
mean_yx2 = np.mean(y_train * x_train**2)

# Constructing the system of equations as matrices
A = np.array([
    [mean_x2, mean_x, 1],
    [mean_x3, mean_x2, mean_x],
    [mean_x4, mean_x3, mean_x2]
])
B = np.array([mean_y, mean_yx, mean_yx2])

# Solve for quadratic regression coefficients
a, b, c = np.linalg.solve(A, B)

# Print quadratic equation coefficients
print("Quadratic Regression Coefficients:")
print("a:", a, "\nb:", b, "\nc:", c)

# Generate predicted values for training and test sets
y_train_pred = a * (x_train ** 2) + b * x_train + c
y_test_pred = a * (x_test ** 2) + b * x_test + c

# Calculate Mean Squared Error (MSE)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("\nMean Squared Error (MSE):")
print("Training Data MSE:", mse_train)
print("Test Data MSE:", mse_test)

# Plot training and test samples
plt.scatter(x_train, y_train, color="blue", label="Training data")
plt.scatter(x_test, y_test, color="green", label="Test data")

# Generate regression curve
x_line_sorted = np.sort(x)
y_line_sorted = a * (x_line_sorted ** 2) + b * x_line_sorted + c
plt.plot(x_line_sorted, y_line_sorted, color='red', linestyle='-', label="Quadratic Regression Curve")

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Quadratic Regression Curve with Training & Test Data")
plt.legend()
plt.grid(True)
plt.show()
