import jax.numpy as np
import jax.random as random
from jax import grad, jit, vmap


class DenseLayer:
    def __init__(self, n_input, n_neuron, activation=lambda x:x, seed=12321):
        self.key = random.PRNGKey(seed)
        self.w = 0.10 * random.normal(self.key, (n_input, n_neuron))
        self.b = np.zeros((1, n_neuron))
        
        self.activation = jit(activation)
        self.calculate_layer = jit(lambda inputs, weights, bias: np.dot(inputs, weights) + bias)

    def forward(self, x):
        self.output =  self.activation(self.calculate_layer(x, self.w, self.b))

    def __call__(self, x):
        self.forward(x)

        return self.output