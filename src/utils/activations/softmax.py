import jax.numpy as np

def softmax(x):
    """
    Description: 
        The softmax activation function.
    Params:
        x: np.ndarray
            The input to the activation function.
    """
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)