import jax.numpy as np
import jax.random as random
from jax import grad, jit, vmap
from ..activations import linear


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
            The activation function to use. (default: linear)
        seed: Int
            The seed to use for the random number generator. (default: 12321)
    """

    def __init__(self, n_input, n_neuron, activation=linear, seed=12321, scale=0.01):
        self.key = random.PRNGKey(seed)
        self.n_input = n_input
        self.n_neuron = n_neuron

        self.w = scale * random.normal(self.key, (n_input, n_neuron))
        self.b = np.zeros((1, n_neuron))

        self.activation = jit(activation)

        self.forward = jit(self.forward)

    def forward(self, x):
        self.output = self.activation(np.dot(x, self.w) + self.b)

        return self.output

    def as_pytree(self):
        return [self.w, self.b]

    def __call__(self, x):
        return self.forward(x)

    def __repr__(self):
        return f"Dense(n_input={self.n_input}, n_neuron={self.n_neuron}, activation={self.activation})"
