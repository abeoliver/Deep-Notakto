# activated.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from agents.Q import Q as BaseQ

class SigmoidHidden (BaseQ):
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
        super(SigmoidHidden, self).__init__(layers, load_file_name, gamma,
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


class SigmoidAll (BaseQ):
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
        super(SigmoidAll, self).__init__(layers, load_file_name, gamma,
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

class TanhHidden (BaseQ):
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
        super(TanhHidden, self).__init__(layers, load_file_name, gamma,
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

class TanhAll (BaseQ):
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
        super(TanhAll, self).__init__(layers, load_file_name, gamma,
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

class SigSoft (BaseQ):
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
        super(SigSoft, self).__init__(layers, load_file_name, gamma,
                                      epsilon, beta, name)
        if name == None:
            self.name = "Q{}_sig_soft_all".format(self.layers)
        else:
            self.name = name

    def _feed(self, inp, n = 0):
        """Recursive function for feeding a vector through layers"""
        # End recursion
        if n == len(self.layers) - 2:
            # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
            return tf.nn.softmax(tf.matmul(inp, self.w[n],
                                           name="feedmul{}".format(n)) + self.b[n])
        # Continue recursion
        out = tf.add(tf.matmul(inp, self.w[n], name="feedmul{}".format(n)), self.b[n],
                     name="feedadd{}".format(n))
        return self._feed(tf.nn.sigmoid(out), n + 1)

class Soft (BaseQ):
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
        super(Soft, self).__init__(layers, load_file_name, gamma,
                                      epsilon, beta, name)
        if name == None:
            self.name = "Q{}_softmax".format(self.layers)
        else:
            self.name = name

    def _feed(self, inp):
        return tf.nn.softmax(self._out(inp))

    def _out(self, inp, n = 0):
        """Recursive function for feeding a vector through layers"""
        # End recursion
        if n == len(self.layers) - 2:
            # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
            return tf.matmul(inp, self.w[n], name="feedmul{}".format(n)) + self.b[n]
        # Continue recursion
        out = tf.add(tf.matmul(inp, self.w[n], name="feedmul{}".format(n)), self.b[n],
                     name="feedadd{}".format(n))
        return self._out(out, n + 1)

    def init_model(self, w = None, b = None):
        """Initialize model"""
        super(Soft, self).init_model(w = w, b = b)
        with self._graph.as_default():
            with tf.name_scope("model"):
                self.out = self._out(self.x)
                # Softmax op
                self.soft_op = tf.placeholder(tf.float64, shape = [self.layers[-1]],
                                              name = "Soft_Op")
                self.softmax = tf.nn.softmax(self.soft_op)

    def get_Q_pre_softmax(self, state):
        """Pass the state to the model and get array of Q-values pre-softmax"""
        return self.out.eval(session = self.session,
                             feed_dict = {self.x: [self.flatten(state)]})[0]

    def target(self, state, action, q, reward, pre_softmax = False, **kwargs):
        """Calculate the target for the network for a given situation"""
        if not pre_softmax:
            return super(Soft).target(state, action, q, reward)
        else:
            # Apply action
            new_state = np.add(state, np.reshape(action, self.shape))
            # Get max Q values after any move opponent could make
            new_Q_max = []
            for move in self.possible_moves(new_state):
                # Make the move
                temp_state = np.add(move, new_state)
                # Find the max Q value
                new_Q_max.append(np.max(self.get_Q_pre_softmax(temp_state)))
            # Get max of all Q values
            maxQ = np.max(new_Q_max)
            # Update Q for target
            Q = np.reshape(self.get_Q_pre_softmax(state), -1)
            Q[np.argmax(action)] = reward + self.gamma * maxQ
            t = self.softmax.eval(session = self.session,
                                  feed_dict = {self.soft_op: Q})
            return np.reshape(t, new_state.shape)