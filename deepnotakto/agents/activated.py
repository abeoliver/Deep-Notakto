# activated.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from agents.Q import Q as BaseQ

class Activated (BaseQ):
    def __init__(self, layers, func, gamma = .8, epsilon = 0.0, beta = None, name = None,
                 initialize = True, **kwargs):
        """
        Initializes an Q learning agent with activations (ABSTRACT)
        Parameters:
            layers (int[]) - Layer architecture for the network
            func (Tensor -> Tensor) - Activation function to apply
            gamma (float [0, 1]) - Q-Learning hyperparameter
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration
            beta (float) - Regularization hyperparameter (if None, regularization
                            is not implemented)
            name (string) - Name of the agent and episodes model
            initialize (bool) - Initialize the model randomly or not
            KWARGS are passed to the model initializer
        Note:
            Initializes randomly if no model is given
        """
        self.func = func
        super(AllActivated, self).__init__(layers, gamma, epsilon, beta, name,
                                           initialize, **kwargs)
        if name == None:
            self.name = "Q{}_activated".format(self.layers)

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

class SigmoidAllActivated (AllActivated):
    def __init__(self, layers, gamma = .8, epsilon = 0.0, beta = 0.1, name = None):
        """
        Initializes an Q learning agent
        Parameters:
            layers (int []) - Number of nodes in each layer
            gamma (float [0, 1]) - Q-Learning hyperparameter
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration
            beta (float) - Regularization hyperparameter
            name (string) - Name of the agent and episodes model
        Note:
            Initializes randomly if no model is given
        """
        super(SigmoidAllActivated, self).__init__(layers, tf.nn.sigmoid, gamma,
                                                  epsilon, beta, name)
        if name == None:
            self.name = "Q{}_sigmoid_all".format(self.layers)

class SigmoidHiddenActivated (HiddenActivated):
    def __init__(self, layers, gamma = .8, epsilon = 0.0, beta = 0.1, name = None):
        """
        Initializes an Q learning agent
        Parameters:
            layers (int []) - Number of nodes in each layer
            gamma (float [0, 1]) - Q-Learning hyperparameter
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration
            beta (float) - Regularization hyperparameter
            name (string) - Name of the agent and episodes model
        Note:
            Initializes randomly if no model is given
        """
        super(SigmoidHiddenActivated, self).__init__(layers, tf.nn.sigmoid, gamma,
                                                     epsilon, beta, name)
        if name == None:
            self.name = "Q{}_sigmoid_hidden".format(self.layers)

class TanhAllActivated (AllActivated):
    def __init__(self, layers, gamma = .8, epsilon = 0.0, beta = 0.1, name = None):
        """
        Initializes an Q learning agent
        Parameters:
            layers (int []) - Number of nodes in each layer
            gamma (float [0, 1]) - Q-Learning hyperparameter
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration
            beta (float) - Regularization hyperparameter
            name (string) - Name of the agent and episodes model
        Note:
            Initializes randomly if no model is given
        """
        super(TanhAllActivated, self).__init__(layers, tf.nn.tanh, gamma,
                                               epsilon, beta, name)
        if name == None:
            self.name = "Q{}_tanh_all".format(self.layers)

class TanhHiddenActivated (HiddenActivated):
    def __init__(self, layers, gamma = .8, epsilon = 0.0, beta = 0.1, name = None):
        """
        Initializes an Q learning agent
        Parameters:
            layers (int []) - Number of nodes in each layer
            gamma (float [0, 1]) - Q-Learning hyperparameter
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration
            beta (float) - Regularization hyperparameter
            name (string) - Name of the agent and episodes model
        Note:
            Initializes randomly if no model is given
        """
        super(TanhHiddenActivated, self).__init__(layers, tf.nn.tanh, gamma,
                                                  epsilon, beta, name)
        if name == None:
            self.name = "Q{}_tanh_hidden".format(self.layers)

class ReluHiddenActivated (HiddenActivated):
    def __init__(self, layers, gamma = .8, epsilon = 0.0, beta = 0.1, name = None):
        """
        Initializes an Q learning agent
        Parameters:
            layers (int []) - Number of nodes in each layer
            gamma (float [0, 1]) - Q-Learning hyperparameter
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration
            beta (float) - Regularization hyperparameter
            name (string) - Name of the agent and episodes model
        Note:
            Initializes randomly if no model is given
        """
        super(ReluHiddenActivated, self).__init__(layers, tf.nn.relu,
                                                  gamma, epsilon, beta, name)
        if name == None:
            self.name = "Q{}_relu_hidden".format(self.layers)

class ReluAllActivated (AllActivated):
    def __init__(self, layers, gamma = .8, epsilon = 0.0, beta = 0.1, name = None):
        """
        Initializes an Q learning agent
        Parameters:
            layers (int []) - Number of nodes in each layer
            gamma (float [0, 1]) - Q-Learning hyperparameter
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration
            beta (float) - Regularization hyperparameter
            name (string) - Name of the agent and episodes model
        Note:
            Initializes randomly if no model is given
        """
        super(ReluAllActivated, self).__init__(layers, tf.nn.relu, gamma,
                                               epsilon, beta, name)
        if name == None:
            self.name = "Q{}_relu_all".format(self.layers)