import jax.numpy as np
import jax.random as random
from jax import grad, jit, vmap
from ..activations import *


class Dense:
    """
    Description: 
        A object that represents a dense layer.
    Params:
        n_input: Int
            The number of input neurons.
        n_neuron: Int
            The number of neurons in the layer.
        activation: callable
            The activation function to use. (default: linear (lambda x:x))
        seed: int
            The seed to use for the random number generator. (default: 12321)
    """

    def __init__(self, n_input, n_neuron, activation=linear, seed=12321):
        self.key = random.PRNGKey(seed)
        self.n_input = n_input
        self.n_neuron = n_neuron

        self.w = 0.10 * random.normal(self.key, (n_input, n_neuron))
        self.b = np.zeros((1, n_neuron))
        
        self.activation = jit(activation)
        self.calculate_layer = jit(lambda inputs, weights, bias: np.dot(inputs, weights) + bias)

        self.forward = jit(self.forward)

    def forward(self, x):
        self.output =  self.activation(self.calculate_layer(x, self.w, self.b))

        return self.output

    def __call__(self, x):
        return self.forward(x)