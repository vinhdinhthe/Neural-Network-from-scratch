import os
import numpy as np

# Load the mnist
def load_mnist(directory):
    # Standard uncompressed MNIST IDX filenames.
    files = {
        'X_train': 'train-images-idx3-ubyte',
        'y_train': 'train-labels-idx1-ubyte',
        'X_test': 't10k-images-idx3-ubyte',
        'y_test': 't10k-labels-idx1-ubyte'
    }
    
    # 1. Load Train Images (Skip 16-byte magic number header)
    with open(os.path.join(directory, files['X_train']), 'rb') as f:
        X_train = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
        
    # 2. Load Test Images (Skip 16-byte magic number header)
    with open(os.path.join(directory, files['X_test']), 'rb') as f:
        X_test = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
        
    # 3. Load Train Labels (Skip 8-byte magic number header)
    with open(os.path.join(directory, files['y_train']), 'rb') as f:
        y_train = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        
    # 4. Load Test Labels (Skip 8-byte magic number header)
    with open(os.path.join(directory, files['y_test']), 'rb') as f:
        y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        
    return X_train, y_train, X_test, y_test

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y]

# Mini batches
def get_mini_batches(X, y, batch_size, shuffle=True):
    """
    Yields slices of X and y arrays as mini-batches.
    """
    # 1. Total number of training samples (60000)
    num_samples = X.shape[0]
    
    # 2. Create an array of indices: [0, 1, 2, ..., 59999]
    indices = np.arange(num_samples)
    
    # 3. Shuffle indices if requested to mix up the digits
    if shuffle:
        np.random.shuffle(indices)
        
    # 4. Loop through the dataset, stepping forward by batch_size
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        
        # Grab the current slice of shuffled indices
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]