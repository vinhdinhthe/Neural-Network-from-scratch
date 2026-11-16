import numpy as np

'''
Include 2 losses function: MSE loss (mean square error) and CE loss
'''

class MSELoss():
    def __call__(self, X, Y):
        return np.mean((X - Y) ** 2) # Batch average

    def grad(self, X, Y):
        return 2 * (X - Y) / X.size

class CrossEntropyLoss():
    def __call__(self, X, Y):
        # X: predicted probabilities, Y: true labels (one-hot encoded)
        eps = 1e-15  # Prevents log(0)
        X = np.clip(X, eps, 1 - eps)
        return -np.sum(Y * np.log(X)) / X.shape[0] # Batch average

    def grad(self, X, Y):
        eps = 1e-15
        X = np.clip(X, eps, 1 - eps)
        return (-Y / X) / X.shape[0]
