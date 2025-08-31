# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def load_data(filename):
    """Loads the dataset from a CSV file."""
    df = pd.read_csv(filename, header=0)  # Skip headers if present
    data = df.iloc[:, :-1].values.astype(float)  # Convert features to NumPy float array
    labels = df.iloc[:, -1].values.astype(int)  # Convert labels to integer array
    return data, labels

def perceptron_learning_algorithm(X, y, max_iter=1000):
    """Implements the Perceptron Learning Algorithm."""
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)  # Initialize weights to zero
    bias = 0
    learning_rate = 1.0

    for _ in range(max_iter):
        errors = 0
        for i in range(num_samples):
            activation = np.dot(weights, X[i]) + bias
            prediction = 1 if activation >= 0 else -1
            if prediction != y[i]:
                weights += learning_rate * y[i] * X[i]
                bias += learning_rate * y[i]
                errors += 1
        if errors == 0:
            break  # Stop if no misclassification
    
    return weights, bias

if __name__ == "__main__":
    filename = r"C:\Users\Dell\Downloads\PLA_Data1 - Sheet1.csv"  # Corrected path
    X, y = load_data(filename)
    weights, bias = perceptron_learning_algorithm(X, y)
    print("Final Weights:", weights)
    print("Final Bias:", bias)

