# activated.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from agents.Q import Q as BaseQ

class QSigmoidHidden (BaseQ):
    def __init__(self, layers, load_file_name = None, gamma = .8,
                 epsilon = 0.0, beta = 0.1, name = None):
        """
        Initializes an Q learning agent with sigmoid on hidden layers
        Parameters:
            size (int) - Board side length
            load_file_name (string) - Path to load saved model from
            gamma (float [0, 1]) - Q-Learning hyperparameter
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration
            beta (float) - Regularization hyperparameter
            name (string) - Name of the agent and its model
        Note:
            Initializes randomly if no model is given
        """
        super(QSigmoidHidden, self).__init__(layers, load_file_name, gamma,
                                       epsilon, beta, name)
        if name == None:
            self.name = "Q{}_sigmoid_hidden".format(self.layers)

    def _feed(self, inp, n = 0):
        """Recursive function for feeding a vector through layers"""
        # End recursion
        if n == len(self.layers) - 2:
            # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
            return tf.matmul(inp, self.w[n], name="feedmul{}".format(n)) + self.b[n]
        # Continue recursion
        out = tf.add(tf.matmul(inp, self.w[n], name="feedmul{}".format(n)), self.b[n],
                     name="feedadd{}".format(n))
        return self._feed(tf.nn.sigmoid(out), n + 1)


class QSigmoidAll (BaseQ):
    def __init__(self, layers, load_file_name = None, gamma = .8,
                 epsilon = 0.0, beta = 0.1, name = None):
        """
        Initializes an Q learning agent with sigmoid on all layers
        Parameters:
            size (int) - Board side length
            load_file_name (string) - Path to load saved model from
            gamma (float [0, 1]) - Q-Learning hyperparameter
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration
            beta (float) - Regularization hyperparameter
            name (string) - Name of the agent and its model
        Note:
            Initializes randomly if no model is given
        """
        super(QSigmoidAll, self).__init__(layers, load_file_name, gamma,
                                       epsilon, beta, name)
        if name == None:
            self.name = "Q{}_sigmoid_all".format(self.layers)

    def _feed(self, inp, n = 0):
        """Recursive function for feeding a vector through layers"""
        # End recursion
        if n == len(self.layers) - 2:
            # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
            return tf.nn.sigmoid(tf.matmul(inp, self.w[n],
                                           name="feedmul{}".format(n)) + self.b[n])
        # Continue recursion
        out = tf.add(tf.matmul(inp, self.w[n], name="feedmul{}".format(n)), self.b[n],
                     name="feedadd{}".format(n))
        return self._feed(tf.nn.sigmoid(out), n + 1)

class QTanhHidden (BaseQ):
    def __init__(self, layers, load_file_name = None, gamma = .8,
                 epsilon = 0.0, beta = 0.1, name = None):
        """
        Initializes an Q learning agent with tanh on all layers
        Parameters:
            size (int) - Board side length
            load_file_name (string) - Path to load saved model from
            gamma (float [0, 1]) - Q-Learning hyperparameter
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration
            beta (float) - Regularization hyperparameter
            name (string) - Name of the agent and its model
        Note:
            Initializes randomly if no model is given
        """
        super(QTanhHidden, self).__init__(layers, load_file_name, gamma,
                                       epsilon, beta, name)
        if name == None:
            self.name = "Q{}_tanh_hidden".format(self.layers)
        else:
            self.name = name

    def _feed(self, inp, n = 0):
        """Recursive function for feeding a vector through layers"""
        # End recursion
        if n == len(self.layers) - 2:
            # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
            return tf.matmul(inp, self.w[n], name="feedmul{}".format(n)) + self.b[n]
        # Continue recursion
        out = tf.add(tf.matmul(inp, self.w[n], name="feedmul{}".format(n)), self.b[n],
                     name="feedadd{}".format(n))
        return self._feed(tf.nn.tanh(out), n + 1)

class QTanhAll (BaseQ):
    def __init__(self, layers, load_file_name = None, gamma = .8,
                 epsilon = 0.0, beta = 0.1, name = None):
        """
        Initializes an Q learning agent with tanh on all layers
        Parameters:
            size (int) - Board side length
            load_file_name (string) - Path to load saved model from
            gamma (float [0, 1]) - Q-Learning hyperparameter
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration
            beta (float) - Regularization hyperparameter
            name (string) - Name of the agent and its model
        Note:
            Initializes randomly if no model is given
        """
        super(QTanhAll, self).__init__(layers, load_file_name, gamma,
                                       epsilon, beta, name)
        if name == None:
            self.name = "Q{}_tanh_all".format(self.layers)
        else:
            self.name = name

    def _feed(self, inp, n = 0):
        """Recursive function for feeding a vector through layers"""
        # End recursion
        if n == len(self.layers) - 2:
            # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
            return tf.nn.tanh(tf.matmul(inp, self.w[n],
                                           name="feedmul{}".format(n)) + self.b[n])
        # Continue recursion
        out = tf.add(tf.matmul(inp, self.w[n], name="feedmul{}".format(n)), self.b[n],
                     name="feedadd{}".format(n))
        return self._feed(tf.nn.tanh(out), n + 1)