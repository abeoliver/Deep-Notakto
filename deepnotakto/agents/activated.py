# activated.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf

from deepnotakto.agents.Q import Q
from deepnotakto.util import unique_classifier


class Activated (Q):
    def __init__(self, layers, func = None, name = None, **kwargs):
        """
        Initializes an Q learning agent with activations (ABSTRACT)
        Parameters:
            layers (int[]) - Layer architecture for the network
            func (Tensor -> Tensor) - Activation function to apply
            name (string) - Name of the agent and episodes model
            KWARGS are passed to the model initializer
        """
        if func == None:
            self.get_func()
        else:
            self.func = func
        super(Activated, self).__init__(layers, name = name, **kwargs)
        if name == None:
            self.get_name()

    def get_name(self):
        self.name = "Q_activated({})".format(unique_classifier())

    def get_func(self):
        self.func = np.identity

    def _feed(self, inp, n = 0):
        pass

class AllActivated (Activated):
    def _feed(self, inp, n = 0):
        """
        Recursively compute func(x.W_i + b_i) for the layers of the network
        Parameters:
            inp ((1, N) array) - Input into the layer
            n (int) - Current layer being applied
        Returns:
            (1, N) array - Output of the given layer (last layer outputs network output)
        """
        # Base case
        if n == len(self.layers) - 2:
            # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
            return self.func(tf.add(tf.matmul(inp, self.w[n], name="feedmul{}".format(n)),
                                    self.b[n], name="feedadd{}".format(n)))
        # Continue recursion
        out = tf.add(tf.matmul(inp, self.w[n], name = "feedmul{}".format(n)), self.b[n],
                     name="feedadd{}".format(n))
        return self._feed(self.func(out), n + 1)

class HiddenActivated (Activated):
    def _feed(self, inp, n = 0):
        """Recursive function for feeding a vector through layers"""
        # End recursion
        if n == len(self.layers) - 2:
            # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
            return tf.add(tf.matmul(inp, self.w[n], name="feedmul{}".format(n)),
                          self.b[n], name = "feedadd{}".format(n))
        # Continue recursion
        out = tf.add(tf.matmul(inp, self.w[n], name = "feedmul{}".format(n)), self.b[n],
                     name="feedadd{}".format(n))
        return self._feed(self.func(out), n + 1)

class SigmoidAll (AllActivated):
    def get_name(self):
        self.name = "Q_sigmoid_all({})".format(unique_classifier())

    def get_func(self):
        self.func = tf.nn.sigmoid

class SigmoidHidden (HiddenActivated):
    def get_name(self):
        self.name = "Q_sigmoid_hidden({})".format(unique_classifier())

    def get_func(self):
        self.func = tf.sigmoid

class TanhAll (AllActivated):
    def get_name(self):
        self.name = "Q_tanh_all({})".format(unique_classifier())

    def get_func(self):
        self.func = tf.nn.tanh

class TanhHidden (HiddenActivated):
    def get_name(self):
        self.name = "Q_tanh_hidden({})".format(unique_classifier())

    def get_func(self):
        self.func = tf.nn.tanh

class ReluHidden (HiddenActivated):
    def get_name(self):
        self.name = "Q_relu_hidden({})".format(unique_classifier())

    def get_func(self):
        self.func = tf.nn.relu

class ReluAll (AllActivated):
    def get_name(self):
        self.name = "Q_relu_all({})".format(unique_classifier())

    def get_func(self):
        self.func = tf.nn.relu