import jax.numpy as np

def step(x):
    return np.where(x > 0, 1, 0)