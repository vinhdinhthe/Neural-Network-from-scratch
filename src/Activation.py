import numpy as np

'''
Include 3 common activation functions: Sigmoid, ReLU, Softmax their forward
prop and backprop diff.
'''

class Sigmoid():
    def __call__(self, X):
        return 1/(1 + np.exp(-X))

    def backward(self, Z, dA):
        # sigmoid'(x) = sigmoid(x) * (1-sigmoid(x))
        S = self(Z)
        return dA * S * (1 - S)


class ReLU():
    def __call__(self, X):
        return np.maximum(0, X)

    def backward(self, Z, dA):
        '''
        ReLU(X) is defined as:
        X > 0 -> ReLU(X) = X -> Diff for X > 0 = 1
        X < 0 -> ReLU(X) = 0 -> Diff for X < 0 = 0
        Or more compacted as: ReLU = max(0, X)
        '''
        return dA * (Z > 0)

class Softmax():
    def __call__(self, X):
        # Subtracting max(X) along the last axis ensures numerical stability
        exp_X = np.exp(X - np.max(X, axis=-1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=-1, keepdims=True)

    def backward(self, Z, dA):
        S = self(Z)
        return S * (dA - np.sum(dA * S, axis=-1, keepdims=True))
