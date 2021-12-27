import jax.numpy as np
from jax import grad, jit, vmap

def categorical_cross_entropy(y_hat, y):
    """
    Description: 
        The categorical cross-entropy loss function for non-batch samples.
    Params:
        y_hat: np.ndarray
            The predicted output.
        y: np.ndarray
            The true output.
    """

    # Clip the predicted values to avoid log(0)
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)

    # Calculate the loss for scalar samples
    if y_hat.ndim == 0:
        return -np.log(y_hat) * y - np.log(1 - y_hat) * (1 - y)

    # Calculate the loss for hot encoded values
    loss = -np.sum(y * np.log(y_hat))
    return loss
