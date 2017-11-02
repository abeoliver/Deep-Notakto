# Q.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from random import choice, sample, shuffle
from agents.agent import Agent
from copy import copy, deepcopy
import pickle, util
import matplotlib.pyplot as plt
import trainer as BaseTrainer

class Q (Agent):
    def __init__(self, layers, gamma = .8, epsilon = 0.0, beta = None, name = None,
                 initialize = True, deterministic = True, classifier = None,
                 training = {"mode": "episodic"}, iterations = 0, **kwargs):
        """
        Initializes an Q learning agent
        Parameters:
            layers (int[]) - Layer architecture for the network
            gamma (float [0, 1]) - Q-Learning hyperparameter
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration
            beta (float) - Regularization hyperparameter (if None, regularization
                            is not implemented)
            name (string) - Name of the agent and episodes model
            initialize (bool) - Initialize the model randomly or not
            deterministic (bool) - Make decisions deterministically or stochastically
            training (dict) - Parameters for training
            KWARGS are passed to the model initializer
        """
        # INITIALIZE
        # Parent initializer
        super(Q, self).__init__(training)
        self.layers = layers
        self.architecture = layers
        self.size = np.int32(np.sqrt(layers[0]))
        self.shape = [self.size, self.size]
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.deterministic = deterministic
        # If classifier is not set, get a new classifier
        if classifier == None:
            self.classifier = util.unique_classifier()
        else:
            self.classifier = classifier
        # If a name is not set, set a default name
        if name == None:
            self.name = "Q({})".format(self.classifier)
        else:
            self.name = name
        # Initialize if desired
        self.initialized = False
        if initialize:
            self.initialize(training = training, iterations = iterations, **kwargs)

    def initialize(self, weights = None, biases = None, force = False,
                   training = None, **kwargs):
        """
        Initialize the model
        Parameters:
            weights (List of (N, M) arrays with variable size) - Initial weights
            biases (List of (1, N) arrays with variable size) - Initial biases
            KWARGS passed to trainer initializer
        """
        if not self.initialized or force:
            # Create a tensorflow session for all processes to run in
            tf.reset_default_graph()
            self._graph = tf.Graph()
            self.session = tf.Session(graph = self._graph)
            # Initialize model
            self.init_model(weights = weights, biases = biases)
            # Initialize training variables like the loss and the optimizer
            self._init_training_vars()
            # Initialize trainer (passing the agent as a parameter)
            self.trainer = QTrainer(self, training = training, **kwargs)
            self.initialized = True

    def init_model(self, weights = None, biases = None):
        """
        Randomly intitialize model, if given weights and biases, treat as a re-initialization
        Parameters:
            weights (List of (N, M) arrays) - Initial weight matricies
            biases (List of (1, N) arrays) - Initial bias matricies
        """
        with self._graph.as_default():
            with tf.name_scope("model"):
                if not self.initialized:
                    s = self.layers[0]
                    self.x = tf.placeholder(tf.float64, shape = [None, s], name = "input")
                    self._init_weights(weights)
                    # Tensorboard visualizations
                    for i, weight in enumerate(self.w):
                        self.variable_summaries(weight, "weight_" + str(i))
                    self._init_biases(biases)
                    # Tensorboard visualizations
                    for i, bias in enumerate(self.b):
                        self.variable_summaries(bias, "bias_" + str(i))
                    # Predicted output
                    self.activation_layers = []
                    self.y = self._feed(self.x)
                    self.initialized = True
                    self.session.run(tf.global_variables_initializer())
                else:
                    if w != None:
                        self.set_weights(weights)
                    if b != None:
                        self.set_biases(biases)

    def _init_training_vars(self):
        """Initialize training procedure"""
        with self._graph.as_default():
            with tf.name_scope("training"):
                # Targets
                self.q_targets = tf.placeholder(shape = [None, self.layers[0]],
                                                 dtype = tf.float64, name = "targets")
                # L2 Regularization
                if self.beta != None:
                    with tf.name_scope("regularization"):
                        self.l2 = self._l2_recurse(self.w)
                        tf.summary.scalar("L2", self.l2)
                # Learning rate
                self.learn_rate = tf.placeholder(tf.float32)
                # Regular loss
                data_loss = tf.reduce_sum(tf.square(self.q_targets - self.y),
                                          name = "data_loss")
                tf.summary.scalar("Data_loss", data_loss)
                # Loss and Regularization
                self.beta_ph = tf.placeholder(tf.float64, name = "beta")
                # L2 Regularization
                if self.beta != None:
                    loss = tf.reduce_mean(tf.add(data_loss, self.beta_ph * self.l2),
                                          name = "regularized_loss")
                else:
                    loss = tf.reduce_mean(data_loss, name = "non_regularized_loss")
                loss = tf.verify_tensor_all_finite(
                    tf.reduce_mean(loss, name = "loss"),
                    msg = "Inf or NaN values",
                    name = "FiniteVerify"
                )
                tf.summary.scalar("Loss", loss)
                # Optimizer
                self._optimizer = tf.train.GradientDescentOptimizer(learning_rate =
                                                                    self.learn_rate,
                                                                    name = "optimizer")
                # Updater (minimizer)
                self.update_op = self._optimizer.minimize(loss, name ="update")
                # Tensorboard
                self.summary_op = tf.summary.merge_all()

    def _init_weights(self, w = None):
        """
        Initialize weights
        Parameters:
            w (List of (N, M) arrays) - Initial weight matricies
        """
        with self._graph.as_default():
            if w != None:
                self.w = [tf.Variable(w[n], name="weights_{}".format(n),
                                      dtype = tf.float64)
                          for n in range(len(self.layers) - 1)]
            else:
                self.w = [tf.Variable(tf.random_normal([self.layers[n],
                                                        self.layers[n + 1]],
                                                       dtype = tf.float64),
                                      name="weights_{}".format(n))
                          for n in range(len(self.layers) - 1)]
            # Get assign opss
            self._weight_assign_ph = [tf.placeholder(tf.float64,
                                                     shape = [self.layers[n],
                                                              self.layers[n + 1]])
                                      for n in range(len(self.layers) - 1)]
            self._weight_assign = [self.w[n].assign(self._weight_assign_ph[n])
                                   for n in range(len(self.layers) - 1)]

    def _init_biases(self, b = None):
        """
        Initialize biases
        Parameters:
            b (List of (1, N) arrays) - Initial bias matricies
        """
        with self._graph.as_default():
            if b != None:
                self.b = [tf.Variable(b[n], name = "biases_{}".format(n),
                                      dtype = tf.float64)
                          for n in range(len(self.layers) - 1)]
            else:
                self.b = [tf.Variable(tf.random_normal([1, self.layers[n + 1]],
                                                       dtype = tf.float64),
                                      name = "biases_{}".format(n))
                          for n in range(len(self.layers) - 1)]
            # Get assign opss
            self._bias_assign_ph = [tf.placeholder(tf.float64,
                                                     shape = [1, self.layers[n + 1]])
                                      for n in range(len(self.layers) - 1)]
            self._bias_assign = [self.b[n].assign(self._bias_assign_ph[n])
                                   for n in range(len(self.layers) - 1)]

    def _feed(self, inp, n = 0):
        """
        Recursively compute x.W_i + b_i for the layers of the network
        Parameters:
            inp ((1, N) array) - Input into the layer
            n (int) - Current layer being applied
        Returns:
            (1, N) array - Output of the given layer (last layer outputs network output)
        """
        # Output of layer
        out = tf.add(tf.matmul(inp, self.w[n], name="feedmul{}".format(n)), self.b[n],
                     name="feedadd{}".format(n))
        # Add to activations list
        self.activation_layers.append(out)
        # Base case (-2 because final layer is output and lists start at zero)
        if n == len(self.layers) - 2:
            return out
        # Continue recursion
        return self._feed(out, n + 1)

    def target(self, state, action, q, reward):
        """
        Calculate the target values for the network in a given situation
        Parameters:
            state ((N, N) array) - Environment state
            action ((N, N) array) - Agents taken action
            q ((N, N) array) - Current Q values
            reward (float) - Scalar reward for the action on the state
        Returns:
            (N, N) array - Target Q matrix for the given state, action, reward pair
        """
        # Apply action
        new_state = np.add(state, action)
        # Get max Q values after any move opponent could make
        new_Q_max = []
        for move in self.action_space(new_state):
            # Make the move
            temp_state = np.add(move, new_state)
            # Find the max Q value
            new_Q_max.append(np.max(self.get_Q(temp_state)))
        # Get max of all Q values
        maxQ = np.max(new_Q_max)
        # Return a new Q vector updated by the Bellman equation
        Q = np.reshape(np.copy(q), -1)
        c = float(Q[np.argmax(action)])
        Q[np.argmax(action)] = reward + max(0.0, self.gamma * maxQ)
        if self.mop:
            print("-------------------")
            print("Action")
            print(action)
            print("Reward        -- {}".format(reward))
            print("Current Value -- {}".format(c))
            print("Future        -- {}".format(maxQ))
            print("Bellman -- {}".format(reward + max(0.0, self.gamma * maxQ)))
        return np.reshape(Q, new_state.shape)

    def get_action(self, state):
        """
        Creates an action vector for a given state
        Returns:
            (N, M) array - An action vector
        """
        # Use e-greedy exploration
        if np.random.rand(1) < self.epsilon:
            action = choice(self.action_space(state))
        elif self.deterministic:
            # Get action Q-vector
            Q = self.get_Q(state)
            # Find the best aciton (largest Q)
            max_index = np.argmax(Q)
            # Make an action vector
            action = np.zeros(Q.size, dtype = np.int32)
            action[max_index] = 1
            action = np.reshape(action, state.shape)
        else:
            pass
        # Return the action
        return action

    def get_Q(self, state):
        """
        Get action Q-values
        Parameters:
            state ((N, N) array) - Current environment state
        Returns:
            (N, N) array - Q matrix for the given state
        """
        # Pass the state to the model and get array of Q-values
        return np.reshape(self.y.eval(session = self.session,
                                      feed_dict = {self.x: [np.reshape(state, -1)]})[0],
                          state.shape)

    def update(self, states, targets, learn_rate = .01, beta = None):
        """
        Update (train) a model over a given set of states and targets
        Parameters:
            states (List of (N, N) arrays) - States to be trained over (inputs)
            targets (List of (N, N) arrays) - Targets to be trained over (labels)
            learn_rate (float) - Learning rate for the update
            beta (float) - L2 Regularization constant (default self.beta)
        Returns:
            tf.summary - The output of the merged summary operation
        """
        # Default to self.beta if no beta is given
        if beta == None:
            beta = self.beta
        # Reshape states and targets of (N, N) to (1, N * N)
        states = np.array([np.reshape(s, -1) for s in states], dtype = np.float32)
        targets = np.array([np.reshape(t, -1) for t in targets], dtype = np.float32)
        # Construct feed dictionary for the optimization step
        feed_dict = {self.x: states, self.q_targets: targets,
                     self.learn_rate: learn_rate, self.beta_ph: beta}
        # Optimize the network and return the tensorboard summary information
        return self.session.run([self.summary_op, self.update_op], feed_dict = feed_dict)[0]

    def save(self, name):
        """
        Save the models parameters in a .npz file
        Parameters:
            name (string) - File name for save file
        """
        # Remove epsilon function from the parameters
        params = copy(self.trainer.params)
        params["epsilon_func"] = None
        with open(name, "wb") as outFile:
            pickle.dump({"weights": [w.eval(session = self.session) for w in self.w],
                         "biases": [b.eval(session = self.session) for b in self.b],
                         "layers": self.layers, "gamma": self.gamma, "name": self.name,
                         "beta": self.beta, "deterministic": self.deterministic,
                         "classifier": self.classifier, "training": params,
                         "iterations": self.trainer.iteration},
                        outFile)

    def variable_summaries(self, var, name):
        """
        Summarize mean/max/min/sd & histogram for TensorBoard visualization
        Parameters:
            var (tf Variable) - Variable to summarize
            name (name) - Name for the variable in TensorBoard
        """
        with tf.name_scope("summary"):
            with tf.name_scope(name):
                tf.summary.scalar('norm', tf.norm(var))
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                # Log var as a histogram
                tf.summary.histogram('histogram', var)

    def get_layer(self, inp, layer):
        """Passes an input through the model to a particular layer"""
        # Pass the state to the model and get array of Q-values
        return self.activation_layers[layer].eval(
            session = self.session,
            feed_dict={self.x: [np.reshape(inp, -1)]})[0]

    def get_node_weights(self, layer, node, reshape = -1):
        """Gets the weight matrix for a given node"""
        return np.reshape(self.get_weight(layer)[:, node], reshape)

    def get_node_activations(self, inp, layer, node):
        """Gets the activation matrix of node at a cerain input"""
        pass

    def get_weight(self, index):
        """Gets an evaluated weight matrix from layer 'index'"""
        return self.w[index].eval(session = self.session)

    def get_weights(self):
        """Gets all weight matricies"""
        return [self.get_weight(i) for i in range(len(self.w))]

    def get_bias(self, index):
        """Gets an evaluated bias matrix from layer 'index'"""
        return self.b[index].eval(session = self.session)[0]

    def get_biases(self):
        """Gets all bias matricies"""
        return [self.get_bias(i) for i in range(len(self.b))]

    def set_weight(self, index, new_w):
        """Replace a given weight with new_w"""
        self.session.run(self._weight_assign[index],
                         feed_dict = {self._weight_assign_ph[index]: new_w})

    def set_bias(self, index, new_b):
        """Replace a given bias with new_b"""
        self.session.run(self._bias_assign[index],
                         feed_dict = {self._bias_assign_ph[index]: new_b})

    def set_weights(self, new_w):
        """Replace all weights with new_w"""
        for i in range(len(self.w)):
            self.set_weight(i, new_w[i])

    def set_biases(self, new_b):
        """Replace all biases with new_b"""
        for i in range(len(self.b)):
            self.set_bias(i, new_b[i])

    def _l2_recurse(self, ws, n = 0):
        """
        Recursively adds all weight norms (actually (norm ^ 2) / 2
        Parameters:
            ws (List of (N, M) arrays) - List of weights to be normed and added
            n (int) - Index for tensor naming
        """
        if len(ws) <= 1:
            return tf.nn.l2_loss(ws[0], name="L2_{}".format(n))
        else:
            return tf.add(tf.nn.l2_loss(ws[0], name="L2_{}".format(n)),
                          self._l2_recurse(ws[1:], n + 1))

    def get_l2(self):
        """Gets the L2 norm of the weights"""
        return self.l2.eval(session = self.session)

    def action_space(self, board):
        """
        Returns a list of all possible moves (reguardless of win / loss)
        Parameters:
            board ((N, N) array) - Current board state
        Returns:
            List of (N, N) arrays - AllActivated legal moves for the given board
        Note:
            An almost identical function exists in the game environment but the
            agent must have an independent mehtod to generate possible moves in
            order to calculate target Q values
        """
        # Get board
        b = copy(board)
        # AllActivated remaining moves
        remaining = []
        # Loop over both axes
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                # If there is an empty space, add the move to remaining moves
                if b[i, j] == 0:
                    z = np.zeros(b.shape, dtype = np.int32)
                    z[i, j] = 1
                    remaining.append(z)
        return remaining

    def train(self, mode = "", **kwargs):
        self.trainer.train(mode, **kwargs)

    def training_params(self, training = None):
        self.trainer.training_params(training)

    def dual(self):
        """
        Creates a shell player whos training affects this agent
        Returns:
            Q Agent
        """
        return Dual(self)

class QTrainer (BaseTrainer.Trainer):
    def default_params(self):
        return {
            "type": "episodic",
            "learn_rate": 1e-4,
            "rotate": False,
            "epsilon_func": None,
            "epochs": 1,
            "batch_size": 1
        }

    def online(self, state, action, reward, learn_rate = None, **kwargs):
        """Gets a callable function for online training"""
        self.offline([state], [action], [reward], 1, 1, learn_rate, **kwargs)

    def offline(self, states = None, actions = None, rewards = None, batch_size = None, epochs = None,
                learn_rate = None, rotate = None):
        """
        Trains the agent over a set of state, action, reward triples
        Parameters:
            states (List of (N, N) arrays) - List of states
            actions (List of (N, N) arrays) - Actions taken on states
            rewards (List of floats) - Rewards for each action
            batch_size (int) - Number of samples in each minibatch
            epochs (int) - Number of iterations over the entire dataset
            learn_rate (float) - Learning rate
            rotate (bool) - Rotate matricies for transformation invariant learning
        Note:
            Targets are caculated at the beginning of each epoch.
            Therefore, all targets in a given epochs use the same
            Q-function and are not effected by the others in the
            batch

            If states, actions, and rewards are None, full history used
        """
        # Default to defaults if parameters not given
        if learn_rate == None:
            learn_rate = self.params["learn_rate"]
        if rotate == None:
            rotate = self.params["rotate"]
        if epochs == None:
            epochs = self.params["epochs"]
        if batch_size == None:
            batch_size = self.params["batch_size"]
        if states == None or actions == None or rewards == None:
            states = self.agent.states
            actions = self.agent.actions
            rewards = self.agent.rewards
        # Train for each epoch
        for epoch in range(epochs):
            # Calculate targets from states, actions, and rewards
            targets = [self.agent.target(state, action, self.agent.get_Q(state), reward)
                       for (state, action, reward) in zip(states, actions, rewards)]
            # Rotate if requested
            if rotate:
                states, targets = self.get_rotations(states, targets)
            # Separate into batches and train
            self.batch(states, targets, batch_size, learn_rate)
        # Change epsilon for e-greedy
        self.change_epsilon()

    def batch(self, states, targets, batch_size, learn_rate = None):
        """
        Trains the agent over a batch of states and targets
        Parameters:
            states (List of (N, N) arrays) - List of states
            targets (List of (N, N) arrays) - List of targets for each state
            batch_size (int) - Number of samples in each minibatch
            learn_rate (float) - Learning rate
        """
        # Default learning rate if none provided
        if learn_rate == None:
            learn_rate = self.learn_rate
        # Batching
        # Shuffle all indicies
        order = list(range(len(states)))
        shuffle(order)
        # Chunk index list into batches of desired size
        batches = list(self.chunk(order, batch_size))
        # Train over each minibatch
        summary = None
        for batch in batches:
            # Get the states and targets for the indicies in the batch and update
            summary = self.agent.update([states[b] for b in batch],
                                         [targets[b] for b in batch],
                                         learn_rate)
        # Record if Tensorboard recording enabled
        if self.record and (self.iteration % self.tensorboard_interval == 0)\
                and summary != None:
            # Write summary to file
            self.writer.add_summary(summary, self.iteration)
        # Increase iteration counter
        self.iteration += 1

    def change_epsilon(self):
        """Changes the epsilon for e-greedy exploration as a function of iteration"""
        if self.params["epsilon_func"] != None:
            self.agent.epsilon = self.params["epsilon_func"](self.iteration)

class Dual (Agent):
    def __init__(self, host):
        super(Dual, self).__init__()
        self.host = host
        self.training = self.host.training
        self.name = host.name + "_dual"

    def get_action(self, state):
        return self.host.get_action(state)

    def get_Q(self, state):
        return self.host.get_Q(state)

    def train(self, mode = ""):
        self.host.train(mode, source_agent = self)