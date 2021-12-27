import jax.numpy as np
from jax import grad, jit, vmap

def calc_accuracy(y_hat, y):
    """
    Description:
        Implement the accuracy metric.
    Params:
        y_hat: np.ndarray
            The predicted output.
        y: np.ndarray
            The true output.
    """
    
    prediction = np.argmax(y_hat, axis=1)
    accuracy = np.mean(prediction == y)

    return accuracy