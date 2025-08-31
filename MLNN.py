import numpy as np
import pandas as pd

# ----------------------------
# Multi-Layer Perceptron Class
# ----------------------------
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, y):
        """Train MLP using backpropagation"""
        y = y.reshape(-1, 1)  # Ensure y is column vector
        for _ in range(self.epochs):
            # Forward pass
            Z1 = np.dot(X, self.W1) + self.b1
            A1 = self.sigmoid(Z1)
            Z2 = np.dot(A1, self.W2) + self.b2
            A2 = self.sigmoid(Z2)

            # Compute error
            error = y - A2

            # Backpropagation
            dA2 = error * self.sigmoid_derivative(A2)
            dW2 = np.dot(A1.T, dA2)
            db2 = np.sum(dA2, axis=0)

            dA1 = np.dot(dA2, self.W2.T) * self.sigmoid_derivative(A1)
            dW1 = np.dot(X.T, dA1)
            db1 = np.sum(dA1, axis=0)

            # Update weights and biases
            self.W2 += self.learning_rate * dW2
            self.b2 += self.learning_rate * db2
            self.W1 += self.learning_rate * dW1
            self.b1 += self.learning_rate * db1

    def predict(self, X):
        """Make predictions"""
        A1 = self.sigmoid(np.dot(X, self.W1) + self.b1)
        A2 = self.sigmoid(np.dot(A1, self.W2) + self.b2)
        return (A2 >= 0.5).astype(int)  # Convert to binary output

    def evaluate(self, X, y):
        """Evaluate accuracy"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y.reshape(-1, 1)) * 100
        return accuracy


# ----------------------------
# Load and Train on User's Dataset
# ----------------------------
file_path =  r"C:\Users\Dell\Downloads\PLA_Data1 - Sheet1.csv"
df = pd.read_csv(file_path)

X_data = df[['x1', 'x2']].values
y_data = df['Class'].values

mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1, epochs=1000)
mlp.fit(X_data, y_data)

accuracy_dataset = mlp.evaluate(X_data, y_data)
print(f"Accuracy on user's dataset: {accuracy_dataset:.2f}%")


# ----------------------------
# Train and Test on Logic Gates (AND, OR, XOR)
# ----------------------------
X_AND = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_AND = np.array([0, 0, 0, 1])

X_OR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_OR = np.array([0, 1, 1, 1])

X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_XOR = np.array([0, 1, 1, 0])

mlp_AND = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1, epochs=10000)
mlp_AND.fit(X_AND, y_AND)
accuracy_AND = mlp_AND.evaluate(X_AND, y_AND)

mlp_OR = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1, epochs=10000)
mlp_OR.fit(X_OR, y_OR)
accuracy_OR = mlp_OR.evaluate(X_OR, y_OR)

mlp_XOR = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1, epochs=10000)
mlp_XOR.fit(X_XOR, y_XOR)
accuracy_XOR = mlp_XOR.evaluate(X_XOR, y_XOR)

print(f"Accuracy on AND gate: {accuracy_AND:.2f}%")
print(f"Accuracy on OR gate: {accuracy_OR:.2f}%")
print(f"Accuracy on XOR gate: {accuracy_XOR:.2f}% (should be 100%)")


# ----------------------------
# Train and Test on IRIS Dataset
# ----------------------------
from sklearn.datasets import load_iris
iris = load_iris()
X_iris = iris.data[:, :2]  # Only using first two features for simplicity
y_iris = (iris.target != 0).astype(int)  # Convert to binary classification (Setosa vs Non-Setosa)

mlp_iris = MLP(input_size=2, hidden_size=6, output_size=1, learning_rate=0.1, epochs=5000)
mlp_iris.fit(X_iris, y_iris)

accuracy_iris = mlp_iris.evaluate(X_iris, y_iris)
print(f"Accuracy on IRIS dataset: {accuracy_iris:.2f}%")