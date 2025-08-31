import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load your data
data = pd.read_csv('MLP_Estimate.csv', sep=',', header=None)

# Features and Target
X = data[[0, 1, 2]].values
y = data[3].values.reshape(-1, 1)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        return self.Z2
    
    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, X, y, y_pred, learning_rate):
        m = X.shape[0]
        
        dZ2 = (y_pred - y) / m
        dW2 = self.A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
    def train(self, X_train, y_train, X_test, y_test, epochs, learning_rate):
        train_errors = []
        test_errors = []
        
        for epoch in range(epochs):
            # Forward
            y_pred_train = self.forward(X_train)
            train_loss = self.compute_loss(y_train, y_pred_train)
            
            # Backward
            self.backward(X_train, y_train, y_pred_train, learning_rate)
            
            # Test forward
            y_pred_test = self.forward(X_test)
            test_loss = self.compute_loss(y_test, y_pred_test)
            
            train_errors.append(train_loss)
            test_errors.append(test_loss)
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}')
        
        return train_errors, test_errors
mlp = MLP(input_size=3, hidden_size=10, output_size=1)

train_errors, test_errors = mlp.train(X_train, y_train, X_test, y_test, epochs=1000, learning_rate=0.01)

plt.plot(train_errors, label='Training Error')
plt.plot(test_errors, label='Testing Error')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training and Testing Errors Over Epochs')
plt.legend()
plt.show()
