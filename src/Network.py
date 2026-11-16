'''
- For the scope of this project, the Network.py is a 3 linear layers
model supports sigmoid / ReLU / softmax activation functions.
- Implementation steps:
    1. Choose widths for 3 linear layers: [w1, w2, w3, ... w{n}]
    2. Random initialization of 3 weight matrices
    3. Forward propagation -> Return results
'''

import numpy as np
if __package__:
    from .Activation import Sigmoid, ReLU, Softmax
else:
    from Activation import Sigmoid, ReLU, Softmax

class NeuralNetwork():
    def __init__(self, seed=6, widths=None, activations = None):
        # widths = [i, h1, ..., h{m}, o]

        if widths is None: widths = [2, 2, 2]
        if len(widths) < 2 or not all(isinstance(x, int) and x > 0 for x in widths):
            raise ValueError("Widths need to contains only positve integers and length > 2")

        if activations is None: activations = ["sigmoid"] * (len(widths) - 1)

        self.rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for n_in, n_out in zip(widths[:-1], widths[1:]):
            W = self.rng.standard_normal((n_in, n_out)) * np.sqrt(2 / (n_in + n_out))
            B = np.zeros((1, n_out))

            self.weights.append(W)
            self.biases.append(B)

        # Initialize activation functions
        activation_map = {
            "sigmoid": Sigmoid,
            "relu": ReLU,
            "softmax": Softmax
        }

        if type(activations) == list and len(activations) != len(widths) - 1:
            raise ValueError("Need one activation per layer transition")
        elif type(activations) != list:
            raise ValueError("Type need to be list")

        if type(activations) == list:
            self.activations = [activation_map[name.lower()]() for name in activations]


    def forward(self, X):
        cache = {}

        A = X  # Input to the first layer is the training data X
        
        for i, (W, B, activation) in enumerate(zip(self.weights, self.biases, self.activations)):
            Z = A @ W + B  # linear transformation
            
            # Save things needed for backprop before 'A' gets updated for the next layer
            cache[i] = {
                'A_prev': A,       # Input to this layer (used for dW)
                'Z': Z,             # Linear output (used for dZ / activation derivative)
                'activation': activation # Storing the function pointer to know how to differentiate it
            }
            
            A = activation(Z)  # Activation function for the current layer

        self.cache = cache

        # Return the final network output and the full cache
        return A 

    def backward(self, dA):
        # dA is gradient from the loss function
        # Retrieve saved cache through self.cache
        num_layers = len(self.weights)

        # Refresh grad cache every backprop
        grad_cache = {
            'dW': [None] * num_layers,
            'dB': [None] * num_layers
        }

        for i in reversed(range(num_layers)):
            layer_cache = self.cache[i]
            A_prev = layer_cache['A_prev']
            Z = layer_cache['Z']
            activation = layer_cache['activation'] # Activation object

            # Compute loss propagated backward through activation function dZ
            dZ = activation.backward(Z, dA) # More general design works for Softmax too instead of dZ = dA * activation.grad(Z)

            # Compute loss for weight and bias, continue backprop
            dW = A_prev.T @ dZ
            # Sum across batch dim - axis = 0, keepDims to prevent collapse to 1D array
            dB = np.sum(dZ, axis=0, keepdims=True) 

            # Save calculated gradients for optimizer to use, so it is like Pytorch backprop processes
            # Though in this we'll just apply simple stochastic gradient descent and will not build separated optimizer
            grad_cache['dW'][i] = dW
            grad_cache['dB'][i] = dB

            # Compute grad dA for PREVIOUS layer i-1
            dA = dZ @ self.weights[i].T

        self.grad_cache = grad_cache

        return

    def step(self, learning_rate):
        # simple stochastic grad descent
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * self.grad_cache['dW'][i]
            self.biases[i] -= learning_rate * self.grad_cache['dB'][i]
        
        return

