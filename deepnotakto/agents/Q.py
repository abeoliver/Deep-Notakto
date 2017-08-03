# Q.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from random import choice, sample, shuffle
from agent import Agent
from copy import copy, deepcopy
import pickle
import matplotlib.pyplot as plt

class Q (Agent):
    def __init__(self, layers, load_file_name = None, gamma = .8,
                epsilon = 0.0, beta = 0.1, name = None):
        """
        Initializes an Q learning agent
        Parameters:
            size (int) - Board side length
            load_file_name (string) - Path to load saved model from
            gamma (float [0, 1]) - Q-Learning hyperparameter (not used if model is loaded)
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration (only when training)
            beta (float) - Regularization hyperparameter
            name (string) - Name of the agent and its model
        Note:
            Initializes randomly if no model is given
        """
        # Call parent initializer
        super(Q, self).__init__()
        self.name = "Q"
        self.layers = layers
        self.size = np.int32(np.sqrt(layers[0]))
        self.shape = [self.size, self.size]
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        if name == None:
            self.name = "Q{}_regular".format(self.layers)
        else:
            self.name = name
        self.rotations = []
        self.initialized = False
        # Create a tensorflow session for all processes to run in
        tf.reset_default_graph()
        self.session = tf.Session()
        # Load model if a file name is given
        if load_file_name != None:
            self.load(load_file_name, "")
        # Otherwise randomly initialize
        else:
            self.init_model()
        # Initialize training variables like the loss and the optimizer
        self.init_training_vars()

    def act(self, env):
        """
        Choose action, apply action to environment, and recieve reward
        Parameters:
            env (environment.Env) - Environment of the agent
        """
        # Current environment state
        current_state = env.observe()
        # Get action Q-vector
        Q = self.get_Q(current_state)
        # Get the action
        # Use e-greedy exploration for added noise
        if np.random.rand(1) < self.epsilon:
            action = choice(env.possible_moves(current_state))
        else:
            action = self.get_action(Q)
        # Apply action, add reward to reward history
        _, reward = env.act(action)

        # Record state, action, reward
        self.record(current_state, action, reward)

        return [current_state, action, reward]

    def target(self, state, action, q, reward):
        """Calculate the target for the network for a given situation"""
        # Apply action
        new_state = np.add(state, np.reshape(action, self.shape))
        # Get max Q values after any move opponent could make
        new_Q_max = []
        for move in self.possible_moves(new_state):
            # Make the move
            temp_state = np.add(move, new_state)
            # Find the max Q value
            new_Q_max.append(np.max(self.get_Q(temp_state)))
        # Get max of all Q values
        maxQ = np.max(new_Q_max)
        # Update Q for target
        Q = np.reshape(q, -1)
        Q[np.argmax(action)] = reward + self.gamma * maxQ
        return np.reshape(Q, new_state.shape)

    def get_action(self, Q):
        """
        Creates an action vector for a given Q-vector
        Parameters:
            Q (N array) - Q-values for a state
            get_index (bool) - Should return max_index or not
        Returns:
            (N, M) array - An action matrix
        """
        # Make a blank action
        action = np.zeros(Q.size, dtype = np.int32)
        # Find and apply the best move
        max_index = np.argmax(Q)
        action[max_index] = 1
        action = np.reshape(action, Q.shape)
        return action

    def get_Q(self, state):
        """
        Get action Q-values
        Parameters:
            state ((N, N) array) - Current environment state
            training (bool) - Should dropout be applied or not
        """
        # Pass the state to the model and get array of Q-values
        return self.y.eval(session = self.session,
                           feed_dict = {self.x: [self.flatten(state)]})[0]

    def init_model(self, w = None, b = None):
        """
        Randomly intitialize model, if given weights and biases, treat as a re-initialization
        """
        with tf.name_scope("model"):
            if (w == None or b == None) or not self.initialized:
                s = self.layers[0]
                self.x = tf.placeholder(tf.float32, shape = [None, s], name = "input")
                # Assign desired values if probided
                if w != None:
                    self.w = [tf.Variable(w[n], name = "weights_{}".format(n))
                              for n in range(len(self.layers) - 1)]
                else:
                    self.w = [tf.Variable(tf.random_normal([self.layers[n],
                                                            self.layers[n + 1]]),
                                          name = "weights_{}".format(n))
                              for n in range(len(self.layers) - 1)]
                if b != None:
                    self.b = [tf.Variable(b[n], name = "biases_{}".format(n))
                              for n in range(len(self.layers) - 1)]
                else:
                    self.b = [tf.Variable(tf.random_normal([1, self.layers[n + 1]]),
                                          name = "biases_{}".format(n))
                              for n in range(len(self.layers) - 1)]
                # Predicted output
                def feed(inp, n=0):
                    """Recursive function for feeding a vector through layers"""
                    # End recursion
                    if n == len(self.layers) - 2:
                        # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
                        return tf.matmul(inp, self.w[n], name = "feedmul{}".format(n)) + self.b[n]
                    # Continue recursion
                    out = tf.add(tf.matmul(inp, self.w[n], name = "feedmul{}".format(n)), self.b[n],
                                 name = "feedadd{}".format(n))
                    return feed(out, n + 1)
                self.y = feed(self.x)
                # Tensorboard visualizations
                for i, weight in enumerate(self.w):
                    self.variable_summaries(weight, "weight" + str(i))
                for i, bias in enumerate(self.b):
                    self.variable_summaries(bias, "Bias" + str(i))
                # Initialize the variables
                self.session.run(tf.global_variables_initializer())
            # If the agent has already been initialized but is being loaded
            elif not (w == None or b == None) and self.initialized:
                print("Parameters loaded from file.")
                for old, new in zip(self.w, w):
                    self.session.run(old.assign(new))
                for old, new in zip(self.b, b):
                    self.session.run(old.assign(new))

    def init_training_vars(self):
        """Initialize training procedure"""
        with tf.name_scope("training"):
            # Targets
            self.q_targets = tf.placeholder(shape = [None, self.layers[0]],
                                             dtype = tf.float32, name = "targets")
            # L2 Regularization
            with tf.name_scope("regularization"):
                def l2_recurse(x, n = 0):
                    """Recursively adds all weight norms"""
                    if len(x) <= 1:
                        return tf.nn.l2_loss(x[0], name = "L2_{}".format(n))
                    else:
                        return tf.add(tf.nn.l2_loss(x[0], name = "L2_{}".format(n)),
                                      l2_recurse(x[1:], n + 1))
                l2 = l2_recurse(self.w)
                tf.summary.scalar("L2", l2)
            # tf.summary.scalar("L2", l2)
            # Learning rate
            self.learn_rate = tf.placeholder(tf.float32)
            # Regular loss
            data_loss = tf.reduce_sum(tf.square(self.q_targets - self.y), name = "data_loss")
            tf.summary.scalar("Data_loss", data_loss)
            # Loss and Regularization
            loss = tf.verify_tensor_all_finite(
                tf.reduce_mean(tf.add(data_loss, tf.constant(self.beta, name = "beta") * l2),
                              name = "loss"),
                msg = "Inf or NaN values",
                name = "FiniteVerify"
            )
            tf.summary.scalar("Loss", loss)
            # tf.summary.scalar("loss", loss)
            # Optimizer
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate =
                                                                self.learn_rate,
                                                                name = "optimizer")
            # Updater (minimizer)
            self.update_op = self._optimizer.minimize(loss, name ="update")
            # Tensorboard
            self.summary_op = tf.summary.merge_all()

    def update(self, states, targets, learn_rate = .01):
        """
        Update (train) a model over a given set of states and targets
        Parameters:
            states (List of (N, N) arrays) - List of states (inputs)
            targets (List of (N, N) arrays) - Targets for each state (labels)
            learn_rate (float) - Learning rate for the update
        Returns:
            tf.summary - The output of the merged summary operation
        """
        # Reshape
        states = np.array([np.reshape(s, -1) for s in states], dtype = np.float32)
        targets = np.array([np.reshape(t, -1) for t in targets], dtype = np.float32)
        feed_dict = {self.x: states, self.q_targets: targets, self.learn_rate: learn_rate}
        return self.session.run([self.summary_op, self.update_op], feed_dict = feed_dict)[0]

    def get_weight(self, index):
        """Gets an evaluated weight matrix from layer 'index'"""
        return self.w[index].eval(session = self.session)

    def get_weights(self):
        """Gets all weight matricies"""
        return [self.get_weight(i) for i in range(len(self.w))]

    def get_bias(self, index):
        """Gets an evaluated bias matrix from layer 'index'"""
        return self.b[index].eval(session = self.session)

    def get_biases(self):
        """Gets all bias matricies"""
        return [self.get_bias(i) for i in range(len(self.b))]

    def save(self, name = None, prefix = "agents/params/"):
        """
        Save the models parameters in a .npz file
        Parameters:
            name (string) - File name for save file
            prefix (string) - The file path prefix
        """
        if not name:
            name = prefix + name + ".npz"
        with open(name, "wb") as outFile:
            pickle.dump({"weights": [w.eval(session = self.session) for w in self.w],
                         "biases": [b.eval(session = self.session) for b in self.b]},
                        outFile)

    def load(self, name, prefix = "agents/params/"):
        """Loads a model from a given file"""
        name = prefix + name
        with open(name, "rb") as inFile:
            loaded = pickle.load(inFile)
            self.init_model(w = loaded["weights"], b = loaded["biases"])

    def possible_moves(self, board):
        """Returns a list of all possible moves (reguardless of win / loss)"""
        # Get board
        b = copy(board)
        # All remaining moves
        remaining = []
        # If in a 2D shape
        if len(b.shape) == 2:
            for i in range(b.shape[0]):
                for j in range(b.shape[1]):
                    if b[i, j] == 0:
                        z = np.zeros(b.shape, dtype = np.int32)
                        z[i, j] = 1
                        remaining.append(z)
        # If in a 1D shape
        else:
            for i in range(b.shape[0]):
                if b[i] == 0:
                    z = np.zeros(b.shape)
                    z[i] = 1
                    remaining.append(z)
        return remaining

    def variable_summaries(self, var, name):
      """Attach mean/max/min/sd & histogram for TensorBoard visualization."""
      with tf.name_scope('summaries'):
        with tf.name_scope(name):
            # Find the mean of the variable
            mean = tf.reduce_mean(var)
            # Log the mean as scalar
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            # Log var as a histogram
            tf.summary.histogram('histogram', var)