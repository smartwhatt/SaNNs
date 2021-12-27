from .categorical_cross_entropy import categorical_cross_entropy
from jax import jit, vmap
import jax.numpy as np

def calc_loss(y_hat, y, loss_fn):
    """
    Description: 
        Calculate the loss.
    Params:
        y_hat: np.ndarray
            The predicted output.
        y: np.ndarray
            The true output.
        loss_fn: function
            The loss function.
    """
    batch_loss = vmap(jit(loss_fn))


    return np.mean(batch_loss(y_hat, y))