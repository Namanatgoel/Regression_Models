import numpy as np

class VectorizedLinearRegression:
    """
    Linear Regression optimized with Gradient Descent using Vectorized Cost Function.
    """
    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray):
        return np.dot(X, self.weights) + self.bias