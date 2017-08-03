# Q.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from random import choice, sample, shuffle
from datetime import datetime
from agent import Agent
from copy import copy, deepcopy
import pickle
import matplotlib.pyplot as plt

class Q (Agent):
    def __init__(self, layers, load_file_name = None, gamma = .8, trainable = True,
                epsilon = 0.0, beta = 0.1, player = 0, sigmoid = False):
        """
        Initializes an Q learning agent
        Parameters:
            size (int) - Board side length
            load_file_name (string) - Path to load saved model from
            gamma (float [0, 1]) - Q-Learning hyperparameter (not used if model is loaded)
            trainable (bool) - Is the model trainable or frozen
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration (only when training)
            beta (float) - Regularization hyperparameter
            player (1, 2) - Player one or two, only used for naming (default 0)
            sigmoid (bool) - Whether to apply the sigmoid activation function to hidden layers
        Note:
            Initializes randomly if no model is given
        """
        # Call parent initializer
        super(Q, self).__init__()
        self.name = "Q"
        self.layers = layers
        self.size = np.int32(np.sqrt(layers[0]))
        self.targets = []
        self.trainable = trainable
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.player = player
        self.sigmoid = sigmoid
        self.train_iteration = 0
        self.name = self.get_name()
        self.rotations = []
        self.initialized = False
        # Create a tensorflow session for all processes to run in
        tf.reset_default_graph()
        self.session = tf.Session()
        # Load model if a file name is given
        if load_file_name != None:
            self.load(load_file_name, "", trainable = trainable)
        # Otherwise randomly initialize
        else:
            self.init_model()
        # Initialize training variables like the loss and the optimizer
        self.init_training_vars()

    def act(self, env, training = False, learn_rate = .0001):
        """
        Choose action, apply action to environment, and recieve reward
        Parameters:
            env (environment.Env) - Environment of the agent
            training (bool) - Should update model with each step or not
            learn_rate (float) - Learning rate for training

        """
        # Current environment state
        current_state = env.observe()
        # Get action Q-vector
        Q = self.get_Q(current_state)
        # Get the action
        # Use e-greedy exploration for added noise
        if np.random.rand(1) < self.epsilon:
            action = choice(env.possible_moves(np.reshape(current_state, -1)))
            action_index = np.argmax(action)
            action = np.reshape(action, env.shape)
        else:
            action, action_index = self.get_action(Q, True)
        # Apply action, add reward to reward history
        new_state, reward = env.act(action)

        # Calculate target for training
        # Get Q values after any move opponent could make
        new_Q_max = []
        for move in env.possible_moves():
            temp_state = np.add(move, new_state)
            new_Q_max.append(np.max(self.get_Q(temp_state)))
        # Max new Q value
        maxQ = np.max(new_Q_max)
        # Update Q for target
        Q[action_index] = reward + self.gamma * maxQ

        # Train if desired
        if training and self.trainable:
            self.update([current_state], [Q], learn_rate = learn_rate)

        # Record state, action, reward, and target
        self.states.append(current_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.targets.append(Q)

        # Change epsilon if training
        # Currently DISABLED
        if training and False:
            self.change_epsilon(self.train_iteration)

    def change_epsilon(self, episode):
        """Changes the epsilon for e-greedy exploration as a function of episode number"""
        self.epsilon = 1.0 / (episode + 1)

    def get_action(self, Q, get_index = False):
        """
        Creates an action vector for a given Q-vector
        Parameters:
            Q (N array) - Q-values for a state
            get_index (bool) - Should return max_index or not
        Returns:
            if not get_index:
                (N, M) array - An action matrix
            else:
                [(N, M) array, int] - Action matrix, max index
        """
        # Make a blank action
        action = np.zeros(Q.shape, dtype = np.int32)
        # Make a stochastic decision
        max_index = np.argmax(Q)
        action[max_index] = 1
        # Add action to action history
        self.actions.append(action)
        # Reshape action for applying
        action = np.reshape(action, [self.size, self.size])
        if get_index:
            return (action, max_index)
        else:
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
                s = self.size * self.size
                self.x = tf.placeholder(tf.float32, shape = [None, s], name = "input")
                # Assign desired values if probided
                if w != None:
                    self.w = [tf.Variable(w[n], trainable = self.trainable,
                                          name = "weights_{}".format(n))
                              for n in range(len(self.layers) - 1)]
                else:
                    self.w = [tf.Variable(tf.random_normal([self.layers[n],
                                                            self.layers[n + 1]]),
                                          trainable = self.trainable,
                                          name = "weights_{}".format(n))
                              for n in range(len(self.layers) - 1)]
                if b != None:
                    self.b = [tf.Variable(b[n], trainable = self.trainable,
                                          name = "biases_{}".format(n))
                              for n in range(len(self.layers) - 1)]
                else:
                    self.b = [tf.Variable(tf.random_normal([1, self.layers[n + 1]]),
                                          trainable = self.trainable,
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
                    if self.sigmoid:
                        return feed(tf.nn.sigmoid(out), n + 1)
                    else:
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
        if not self.trainable:
            return None
        with tf.name_scope("training"):
            # Targets
            self.q_targets = tf.placeholder(shape = [None, self.size * self.size],
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
            self._update_op = self._optimizer.minimize(loss, name = "update")

            # Tensorboard and debug
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter("tensorboard/" + self.name, self.session.graph)

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

    def update(self, states, targets, learn_rate = .01):
        """Update (train) a model over a given set of states and targets"""
        if not self.trainable:
            return None
        # Reshape
        states = np.array([np.reshape(s, -1) for s in states], dtype = np.float32)
        targets = np.array([np.reshape(t, -1) for t in targets], dtype = np.float32)
        # Run training update
        self.train_iteration += 1
        feed_dict = {self.x: states, self.q_targets: targets, self.learn_rate: learn_rate}
        summary, _ = self.session.run([self.merged, self._update_op], feed_dict = feed_dict)
        self.writer.add_summary(summary, self.train_iteration)

    def train(self, batch_size, epochs, learn_rate = .01, rotate_all = True,
              states = [], targets = []):
        """
        Trains the model over entire history
        Parameters:
            batch_size (int) - Number of samples in each minibatch
            epochs (int) - Number of iterations over the entire dataset
            rotate_all (bool) - Add rotations into the dataset
            states (List of (N, N) arrays) - List of states to train over
            targets (List of (N, N) arrays) - List of targets to train over
        Note:
            If states or targets are empty (or don't have matching dimensions), then full
            state and target histories are used
        """
        # Get state and target lists
        if not (len(states) > 0 and len(targets) > 0 and len(states) == len(targets)):
            states = copy(self.states)
            targets = copy(self.targets)
        # Get rotations if requested
        if rotate_all:
            pass
        # Output
        print("Training ", end = "")
        display_interval = epochs // 10 if epochs >= 10 else 1
        for epoch in range(epochs):
            # Batching
            # Shuffle all indicies
            order = list(range(len(states)))
            shuffle(order)
            # Chunk into batches
            batches = list(self._chunk(order, batch_size))
            for b in batches:
                # Get the states and targets from the indidicies of the batch and pass into update
                self.update([states[i] for i in b],
                            [targets[i] for i in b],
                            learn_rate)
            # Output
            if epoch % display_interval == 0:
                print("*", end = "")
        print(" Done")

    def _chunk(self, l, n):
        """
        Yield successive n-sized chunks from l
        Taken from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        """
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def get_rotations(self, states, targets):
        """Train over the rotated versions of each state and reward"""
        # Copy states so as not to edit outside of scope
        states = copy(states)
        # Reshape targets for rotation
        targets = [np.reshape(t, states[0].shape) for t in targets]
        # Collect rotated versions of each state and target
        new_tates = []
        new_targets = []
        print("Rotating ... ", end = "")
        for s, t in zip(states, targets):
            ns = s
            nt = t
            for i in range(3):
                rs = self.rotate(ns)
                rt = self.rotate(nt)
                new_states.append(rs)
                new_targets.append(np.reshape(rt, -1))
                ns = rs
                nt = rt
        print("Done")
        # Combine lists
        all_states = states + new_states
        all_targets = targets + new_targets
        return [allStates, allTargets]

    def rotate(self, x):
        """Rotates an array counter-clockwise"""
        n = np.zeros(x.shape)
        for i in range(x.shape[0]):
            n[:, i] = x[i][::-1]
        return n

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

    def get_name(self):
        """Gets a model name based on the current date and time"""
        today = datetime.now()
        return "Q_P{0}_{1}_{2}_{3}_{4}_{5}_{6}".format(
            self.player, self.layers, str(today.year)[2:], today.month, today.day, today.hour, today.minute)

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

    def load(self, name, prefix = "agents/params/", trainable = False):
        """Loads a model from a given file"""
        name = prefix + name
        with open(name, "rb") as inFile:
            loaded = pickle.load(inFile)
            self.trainable = trainable
            self.init_model(w = loaded["weights"], b = loaded["biases"])

    def show_Q(self, board):
        """Visualize the Q values"""
        plt.imshow(np.reshape(self.get_Q(board), [3, 3]))
        plt.show()
        return ""

    def reset_memory(self):
        """Reset the memory of an agent"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.targets = []