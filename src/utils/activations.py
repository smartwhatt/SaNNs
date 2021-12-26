import jax.numpy as np
from jax import grad, jit, vmap


def linear(x):
    return x

def sigmiod(x):
    return 1/(1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)