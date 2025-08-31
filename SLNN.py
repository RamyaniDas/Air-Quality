import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation(self, x):
        """Step activation function"""
        return np.where(x >= 0, 1, -1)

    def fit(self, X, y):
        """Train the perceptron on given dataset X and labels y"""
        num_features = X.shape[1]
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.epochs):
            for i in range(X.shape[0]):
                # Compute weighted sum
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(linear_output)

                # Update weights if prediction is incorrect
                if y_pred != y[i]:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]

    def predict(self, X):
        """Make predictions on input data X"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation(linear_output)

    def evaluate(self, X, y):
        """Evaluate accuracy on dataset"""
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y) * 100
        return accuracy


# --------------------------
# Load and Train on User's Dataset
# --------------------------
file_path = r"C:\Users\Dell\Downloads\PLA_Data1 - Sheet1.csv"
df = pd.read_csv(file_path)

X_data = df[['x1', 'x2']].values
y_data = df['Class'].values

# Train the perceptron on the provided dataset
perceptron = Perceptron(learning_rate=0.1, epochs=100)
perceptron.fit(X_data, y_data)

# Evaluate performance
accuracy_dataset = perceptron.evaluate(X_data, y_data)
print(f"Accuracy on user's dataset: {accuracy_dataset:.2f}%")

# --------------------------
# Train and Test on Logic Gates (AND, OR, XOR)
# --------------------------
X_AND = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_AND = np.array([-1, -1, -1, 1])  # AND logic

X_OR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_OR = np.array([-1, 1, 1, 1])  # OR logic

X_XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_XOR = np.array([-1, 1, 1, -1])  # XOR logic (not linearly separable)

# Train and evaluate perceptron for each logic function
perceptron_AND = Perceptron(learning_rate=0.1, epochs=10)
perceptron_AND.fit(X_AND, y_AND)
accuracy_AND = perceptron_AND.evaluate(X_AND, y_AND)

perceptron_OR = Perceptron(learning_rate=0.1, epochs=10)
perceptron_OR.fit(X_OR, y_OR)
accuracy_OR = perceptron_OR.evaluate(X_OR, y_OR)

perceptron_XOR = Perceptron(learning_rate=0.1, epochs=10)
perceptron_XOR.fit(X_XOR, y_XOR)
accuracy_XOR = perceptron_XOR.evaluate(X_XOR, y_XOR)

print(f"Accuracy on AND gate: {accuracy_AND:.2f}%")
print(f"Accuracy on OR gate: {accuracy_OR:.2f}%")
print(f"Accuracy on XOR gate: {accuracy_XOR:.2f}% (expected failure)")
