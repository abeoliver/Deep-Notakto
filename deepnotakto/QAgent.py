#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: agents/Q.py                                                  #
#  Abraham Oliver, 2018                                               #
#######################################################################

# Import dependencies
import pickle
from random import choice, shuffle

import numpy as np
import tensorflow as tf

import deepnotakto.util as util
from deepnotakto import Agent
from deepnotakto import Trainer


def get_activation_function(func):
    """ Get a tensorflow-based activation function by name """
    func = func.lower()
    if func == "sigmoid":
        return tf.sigmoid
    elif func == "tanh":
        return tf.nn.tanh
    elif func == "relu":
        return tf.nn.relu
    elif func == "swish":
        return lambda x: tf.multiply(x, tf.sigmoid(x))
    else:
        return tf.identity


class QAgent (Agent):
    """
    A Deep-Q agent

    Methods:
        initialize - Initialize model and tensorflow-specific attributes
        target - Calculate the target policy for a given data point
        get_action - Get the move vector to play on a given state
        get_q - Get a Q-value matrix (a square policy vector)
        get_passable_state - Clean a given state and return a valid model input
        update - Run a training update over given training data
        duplicative_dict - Get a dict that is sufficient to replicate the agent
        save - Save the agent as a duplicative dict with a given name
        copy - Create a copy of the agent
        variable_summaries - Add tensorboard nodes for stats of a given var
        get_layer - Get the output of desired layer instead of the final output
        get_node_weights - Get the weight matrix for a given node
        get_weights - Get all weight parameters
        get_biases - Get all bias parameters
        get_l2 - Get the L2 norm of the weight parameters
        action_space - Get list of all possible moves
        train - Use the trainer to train the model
        training_params - Change the training parameter dictionary
        change_param - Change an individual training parameter
    """
    def __init__(self, game_size, layers, gamma = .8, beta = None,
                 name = None, initialize = True, classifier = None,
                 iterations = 0, params = {"mode": "replay"}, max_queue = 100,
                 epsilon_func = None, temp_func = None,
                 activation_func = "identity", activation_type = "hidden",
                 **kwargs):
        """
        Initialize a Q learning agent

        Args:
            game_size: (int or int[]) Side length of the board
            hidden_layers: (int[]) Number of neurons in each hidden layer
            gamma: (float) Q-learning bellman equation hyperparameter
            beta: (float) L2 regularization hyperparameter
            name: (string) Agent name
            initialize: (bool) Run model and tensorflow initializer?
            classifier: (string) Unique identifier for agent
            iterations: (int) Current iteration of model
            params: (dict) Training parameters
            max_queue: (int) Maximum length of the memory queue
            epsilon_func: (int -> float) Scheduler for exploration constant
            temp_func: (float) (int -> float) Scheduler for temperature
            activation_func: (string) Activaion func to be used on neurons
            activation_type: ("all" or "hidden") Neurons to apply activation to
            kwargs: (dict) Arguments to pass to trainer
        """
        # Parent initializer
        super(QAgent, self).__init__(params, max_queue = max_queue)
        # The length of the side of the board
        self.size = game_size
        # Put together the layers
        self.layers = layers
        self.gamma = gamma
        self.beta = beta
        self.clip_thresh = 10.0
        # Get the activation function
        self.activation_func_name = activation_func
        self.activation_func = get_activation_function(activation_func)
        self.activation_type = activation_type
        # Decide if the agent is deterministic or not
        if epsilon_func is None and temp_func is None:
            self.deterministic = True
        else:
            self.deterministic = False
        # Clean the exploration epsilon function
        if epsilon_func is None:
            self._epsilon_func = epsilon_func
        elif type(epsilon_func) in [float, int]:
            self._epsilon_func = lambda x: epsilon_func
        elif isinstance(epsilon_func, type(lambda: None)):
            self._epsilon_func = epsilon_func
        else:
            raise ValueError("Not a valid epsilon scheduler")
        # Clean the temperature scheduler
        if temp_func is None:
            self._temp_func = temp_func
        elif type(temp_func) in [float, int]:
            self._temp_func = lambda x: temp_func
        elif isinstance(temp_func, type(lambda: None)):
            self._temp_func = temp_func
        else:
            raise ValueError("Not a valid temperature scheduler")
        # If classifier is not set, get a new classifier
        if classifier is None:
            self.classifier = util.unique_classifier()
        else:
            self.classifier = classifier
        # If a name is not set, set a default name
        if name is None:
            self.name = "Q({})".format(self.classifier)
        else:
            self.name = name
        # Initialize if desired
        self.initialized = False
        if initialize:
            self.initialize(params = params, iterations = iterations, **kwargs)

    def initialize(self, weights = None, biases = None, force = False,
                   params = None, **kwargs):
        """
        Initialize the model

        Args:
            weights: (List of (N, M) arrays with variable size) Initial weights
            biases: (List of (1, N) arrays with variable size) Initial biases
            force: (bool) Initialize, even if already initialized
            params: (dict) Training parameters
            KWARGS passed to trainer initializer
        """
        if not self.initialized or force:
            # Create a tensorflow session for all processes to run in
            self._graph = tf.Graph()
            self.session = tf.compat.v1.Session(graph = self._graph)
            # Initialize model
            self._init_model(weights = weights, biases = biases)
            # Initialize params variables like the loss and the optimizer
            self._init_training_vars()
            # Initialize trainer (passing the agent as a parameter)
            self._init_trainer(params = params, **kwargs)
            self.initialized = True

    def _init_trainer(self, params = None, **kwargs):
        """ Initialize a trainer object """
        self.trainer = QTrainer(self, params = params, **kwargs)

    def _init_model(self, weights = None, biases = None):
        """
        Randomly intitialize model, if given parameters, treat as a re-init

        Args:
            weights: (List of (N, M) arrays) Initial weight matricies
            biases: (List of (1, N) arrays) Initial bias matricies
        """
        with self._graph.as_default():
            with tf.name_scope("model"):
                if not self.initialized:
                    # Input placeholder
                    self.x = tf.compat.v1.placeholder(
                        tf.float32, shape = [None, self.layers[0]],
                        name = "inputs")
                    # Initialize weights
                    self._init_weights(weights)
                    # Tensorboard visualizations for weight variables
                    for i, weight in enumerate(self.w):
                        self.variable_summaries(weight, "weight_" + str(i))
                    # Initialize biases
                    self._init_biases(biases)
                    # Tensorboard visualizations for bias variables
                    for i, bias in enumerate(self.b):
                        self.variable_summaries(bias, "bias_" + str(i))
                    # Prepare a list for individual layer outputs
                    self.activation_layers = []
                    # Output prediction vector
                    self.y = self._feed(self.x)
                    self.initialized = True
                    self.session.run(tf.compat.v1.global_variables_initializer())
                # Set node values instead of creating nodes if already init
                else:
                    if w is not None:
                        self.set_weights(weights)
                    if b is not None:
                        self.set_biases(biases)

    def _init_training_vars(self):
        """ Initialize training procedure """
        with self._graph.as_default():
            # Target placeholder
            self.q_targets = tf.compat.v1.placeholder(
                tf.float32, shape = [None, self.layers[0]], name = "q_targets"
            )
            # Learning rate
            self.learn_rate = tf.compat.v1.placeholder(tf.float32,
                                                       name = "learn_rate")
            # Loss
            self._loss = self._get_loss_function()
            # Optimizer
            self._optimizer = tf.compat.v1.train.GradientDescentOptimizer(
                learning_rate = self.learn_rate, name = "optimizer")
            # Get gradients
            self._gradients = self._optimizer.compute_gradients(self._loss)
            # Verify finite and real and save in (gradient, variable) pairs
            name = "FiniteGradientVerify"
            grads = [(
                         tf.compat.v1.verify_tensor_all_finite(
                             g,
                             msg = "Inf or NaN Gradients for {}".format(v.name),
                             name = name),
                         v
                     )
                     for g, v in self._gradients]
            # Clip gradients and save in (gradient, variable) pairs
            self._clipping_threshold = tf.compat.v1.placeholder(
                tf.float32, name = "grad_clip_thresh"
            )
            self._clipped_gradients = [(
                                           tf.clip_by_norm(
                                               g, self._clipping_threshold),
                                           v
                                       )
                                       for g, v in grads]
            # Updater (minimizer)
            self.update_op = self._optimizer.apply_gradients(
                self._clipped_gradients, name = "update")
            # Tensorboard
            self.summary_op = tf.compat.v1.summary.merge_all()

    def _get_loss_function(self):
        """ Get the loss function """
        with tf.name_scope("loss"):
            # Regular loss
            data_loss = tf.reduce_sum(tf.square(self.q_targets - self.y),
                                      name="data_loss")
            tf.compat.v1.summary.scalar("Data_loss", data_loss)
            # Loss and Regularization
            self.beta_ph = tf.compat.v1.placeholder(tf.float32, name="beta")
            # L2 Regularization (if turned on by a non-None beta)
            if self.beta is not None:
                with tf.name_scope("regularization"):
                    self.l2 = self._l2_recurse(self.w)
                    tf.compat.v1.summary.scalar("L2", self.l2)
                    loss = tf.reduce_mean(
                        tf.add(data_loss, self.beta_ph * self.l2),
                        name = "regularized_loss")
            # Otherwise do not a regularization term
            else:
                loss = tf.reduce_mean(data_loss, name = "non_regularized_loss")
            # Verify that the loss is finite and not NaN
            loss = tf.compat.v1.verify_tensor_all_finite(
                tf.reduce_mean(loss, name = "loss"),
                msg = "Inf or NaN loss",
                name = "FiniteVerify"
            )
            # Save to tensorboard summary
            tf.compat.v1.summary.scalar("Loss", loss)
            return loss

    def _init_weights(self, w = None):
        """
        Initialize weight matricies either randomly or with a given set

        Parameters:
            w: (array[]) Initial weight matricies
        """
        # If weights are supplied, use those
        if w is not None:
            self.w = [tf.Variable(w[n], name = "weights_{}".format(n),
                                  dtype = tf.float32)
                      for n in range(len(self.layers) - 1)]
        # Otherwise, initialize with a normal distribution
        else:
            self.w = [tf.Variable(tf.random.normal([self.layers[n],
                                                    self.layers[n + 1]],
                                                   stddev = .2),
                                  name = "weights_{}".format(n))
                      for n in range(len(self.layers) - 1)]
        # Get assignment operations
        self._weight_assign_ph = [tf.compat.v1.placeholder(
            tf.float32, shape = [self.layers[n], self.layers[n + 1]]
        )
            for n in range(len(self.layers) - 1)]
        self._weight_assign = [self.w[n].assign(self._weight_assign_ph[n])
                               for n in range(len(self.layers) - 1)]

    def _init_biases(self, b = None):
        """
        Initialize biases

        Args:
            b: (array[]) Initial bias matricies
        """
        # If supplied, use given
        if b is not None:
            self.b = [tf.Variable(b[n], name = "biases_{}".format(n),
                                  dtype = tf.float32)
                      for n in range(len(self.layers) - 1)]
        # Otherwise, initialize with zeros
        else:
            self.b = [tf.Variable(tf.zeros([1, self.layers[n + 1]]),
                                  name = "biases_{}".format(n))
                      for n in range(len(self.layers) - 1)]
        # Get assign ops
        self._bias_assign_ph = [tf.compat.v1.placeholder(
            tf.float32, shape = [1, self.layers[n + 1]]
        )
            for n in range(len(self.layers) - 1)]
        self._bias_assign = [self.b[n].assign(self._bias_assign_ph[n])
                             for n in range(len(self.layers) - 1)]

    def _feed(self, inp, n = 0):
        """
        Recursively compute x_i.W_i + b_i for the layers of the network

        Args:
            inp: (array) Input into the layer
            n: (int) Current layer being calculated
        Returns:
            (array) Output of the given layer
        """
        # Output of layer
        out = tf.add(tf.matmul(inp, self.w[n], name = "feedmul{}".format(n)),
                     self.b[n],
                     name = "feedadd{}".format(n))
        # Base case (-2 because final layer is output and lists start at zero)
        if n == len(self.layers) - 2:
            if self.activation_type.lower() == "all":
                out = self.activation_func(out)
            self.activation_layers.append(out)
            return out
        # Add to activations list
        self.activation_layers.append(out)
        # Continue recursion
        return self._feed(self.activation_func(out), n + 1)

    def target(self, state, action, reward):
        """
        Calculate the target values for the network in a given situation

        Args:
            state: (array) Environment state
            action: (array) Agents taken action
            reward: (float) Scalar reward for the action on the state
        Returns:
            array - Target Q matrix for the given data
        """
        # Apply action
        new_state = np.add(state, action)
        # Get current Q values
        q = np.reshape(np.copy(self.get_q(state)), -1)
        # If the game is over, return a new Q updated with the observed reward
        if self.is_over(new_state):
            q[np.argmax(action)] = reward
        # Otherwise, use the bellman equation and the next set of possible moves
        else:
            # Get max Q values after any move opponent could make
            new_q_max = []
            for move in self.action_space(new_state):
                # Make the move
                temp_state = np.add(move, new_state)
                # Find the max Q value
                new_q_max.append(np.max(self.get_q(temp_state)))
            # Get max of all Q values
            max_q = np.max(new_q_max)
            # Return a new Q vector updated by the Bellman equation
            q[np.argmax(action)] = reward + self.gamma * max_q
        return np.reshape(q, new_state.shape)

    def get_action(self, state):
        """
        Creates an action vector for a given state

        Args:
            state: (array) Environment state
        Returns:
            (array) An action vector
        """
        # Get Q-values
        qs = self.get_q(state)
        # Use softmax selection if requested
        if not self.deterministic and self.temperature > .01:
            probs = util.softmax(np.reshape(qs, -1) / self.temperature)
            if not np.isnan(probs).any():
                # Get the randomly chosen action
                action = np.zeros(state.size)
                action[np.random.choice(state.size, p = probs)] = 1
                return np.reshape(action, state.shape)
        # If softmax not requested or temperature too low
        # Use e-greedy exploration
        if not self.deterministic and np.random.rand(1) < self.epsilon:
            return choice(self.action_space(state))
        # Use max-value selection
        else:
            # Find the best aciton (largest Q)
            max_index = np.argmax(qs)
            # Make an action vector
            action = np.zeros(qs.size, dtype = np.int32)
            action[max_index] = 1
            return np.reshape(action, state.shape)

    def get_q(self, state):
        """
        Get action Q-values

        Args:
            state: (array) Current environment state
        Returns:
            (array) Q matrix for the given state
        """
        # Pass the state to the model and get array of Q-values
        return np.reshape(
            self.y.eval(session = self.session,
                        feed_dict = {
                            self.x: self.get_passable_state(state)
                        })[0],
            state.shape)

    def get_passable_state(self, state):
        """
        Prepare a valid input for the computation graph of the model

        Args:
            state: (array or array[]) State or states to clean for network feed
        Returns:
            (array or array[]) State or states fit to be passed through network
        """
        return [np.reshape(state, -1)]

    def update(self, states, targets, learn_rate = .01, beta = None):
        """
        Update (train) a model over a given set of states and targets

        Args:
            states (array[]) - States to be trained over (inputs)
            targets (array[]) - Targets to be trained over (labels)
            learn_rate (float) - Learning rate for the update
            beta (float) - L2 Regularization constant (default self.beta)
        Returns:
            (tf.summary) The output of the merged summary operation
        """
        # Reshape states and targets of (N, N) to (1, N * N)
        states_ = np.array([np.reshape(s, -1) for s in states],
                           dtype = np.float32)
        targets_ = np.array([np.reshape(t, -1) for t in targets],
                            dtype = np.float32)
        # Default to self.beta if no beta is given
        if beta is None:
            beta = self.beta
        # Construct feed dictionary for the optimization step
        feed_dict = {self.x: states_, self.q_targets: targets_,
                     self.learn_rate: learn_rate, self.beta_ph: beta,
                     self._clipping_threshold: self.clip_thresh}
        # Optimize the network and return the tensorboard summary information
        summary = self.session.run([self.summary_op, self.update_op],
                                   feed_dict = feed_dict)[0]
        return summary

    def duplicative_dict(self):
        """ Saveable dictionary able to replicate the agent """
        return {"game_size": self.size, "layers": self.layers,
                "weights": self.get_weights(), "biases": self.get_biases(),
                "gamma": self.gamma, "name": self.name,
                "beta": self.beta, "classifier": self.classifier,
                "params": self.params, "iterations": self.iteration,
                "max_queue": self.max_queue,
                "tensorboard_interval": self.trainer.tensorboard_interval,
                "tensorboard_path": self.trainer.tensorboard_path,
                "activation_func": self.activation_func_name,
                "activation_type": self.activation_type}

    def save(self, name):
        """
        Save the models parameters in a .npz file

        Args:
            name: (string) File name for save file
        """
        with open(name, "wb") as outFile:
            pickle.dump(self.duplicative_dict(), outFile)

    def copy(self):
        """ Create a perfect replica of the agent """
        return self.__class__(**self.duplicative_dict())

    def variable_summaries(self, var, name):
        """
        Summarize mean, max, min, sd, histogram for TensorBoard visualization

        Args:
            var: (tf.Variable) Variable to summarize
            name: (name) Name for the variable in TensorBoard
        """
        with tf.name_scope("summary"):
            with tf.name_scope(name):
                tf.compat.v1.summary.scalar('norm', tf.norm(var))
                tf.compat.v1.summary.scalar('max', tf.reduce_max(var))
                tf.compat.v1.summary.scalar('min', tf.reduce_min(var))
                # Log var as a histogram
                tf.compat.v1.summary.histogram('histogram', var)

    def get_layer(self, inp, layer):
        """ Passes an input through the model until a particular layer """
        # Pass the state to the model and get array of Q-values
        return self.activation_layers[layer].eval(
            session = self.session,
            feed_dict={self.x: [np.reshape(inp, -1)]})[0]

    def get_node_weights(self, layer, node, reshape = -1):
        """ Gets the weight matrix for a given node """
        return np.reshape(self.get_weight(layer)[:, node], reshape)

    def get_weights(self, index = None):
        """ Gets all weight matricies """
        if index is None:
            return [self.w[i].eval(session = self.session)
                    for i in range(len(self.w))]
        return self.w[index].eval(session = self.session)

    def get_biases(self, index = None):
        """ Gets all bias matricies """
        if index is None:
            return [self.b[i].eval(session = self.session)
                    for i in range(len(self.b))]
        return self.b[index].eval(session = self.session)

    def set_weights(self, new_w, index = None):
        """
        Replace either all weights or a given weight layer with new parameters

        Args:
            new_w: (array or array[]) New weights to be used
            index: (int or None) Desired weight layer or None for all layers
        """
        def set_weight(obj, i, w):
            if w.shape == obj.w[i].shape:
                obj.session.run(obj._weight_assign[i],
                                feed_dict = {obj._weight_assign_ph[i]: w})
            else:
                raise (ValueError("Shape for weight #{} must be {}".format(
                    i, obj.w[i].shape)))
        if index is None:
            for i in range(len(self.w)):
                set_weight(self, i, new_w[i])
        else:
            set_weight(self, index, new_w)

    def set_biases(self, new_b, index = None):
        """
        Replace either all biases or a given bias layer with new parameters

        Args:
            new_b: (array or array[]) New biases to be used
            index: (int or None) Desired bias layer or None for all layers
        """
        def set_bias(obj, i, b):
            if b.shape == obj.b[i].shape:
                obj.session.run(obj._bias_assign[i],
                                feed_dict = {obj._bias_assign_ph[i]: w})
            else:
                raise (ValueError("Shape for bias #{} must be {}".format(
                    i, obj.w[i].shape)))
        if index is None:
            for i in range(len(self.b)):
                self.set_bias(i, new_b[i])
        else:
            set_bias(self, index, new_b)

    def _l2_recurse(self, ws, n = 0):
        """
        Recursively adds all weight norms

        Args:
            ws: (arrays[]) List of weights to be normed and added
            n: (int) Index for tensor naming
        Returns:
            (float) Total L2 norm
        """
        if len(ws) <= 1:
            return tf.nn.l2_loss(ws[0], name="L2_{}".format(n))
        else:
            return tf.add(tf.nn.l2_loss(ws[0], name="L2_{}".format(n)),
                          self._l2_recurse(ws[1:], n + 1))

    def get_l2(self):
        """ Gets the L2 norm of the weights """
        return self.l2.eval(session = self.session)

    def action_space(self, state):
        """
        Returns a list of all possible moves (reguardless of win / loss)

        Args:
            state: (array) Current board state
        Returns:
            (array[]) - All legal moves for the given board
        """
        return []

    def train(self, mode = "", **kwargs):
        """ Train the model using the trainer """
        self.trainer.train(mode, **kwargs)

    def training_params(self, training = None):
        """ Use new training parameter dictionary """
        self.trainer.training_params(training)

    def change_param(self, name, value):
        """ Change a given training parameter"""
        self.trainer.change_param(name, value)

    @property
    def epsilon(self):
        if self._epsilon_func is None:
            return 0
        return self._epsilon_func(self.iteration)

    @epsilon.setter
    def epsilon(self, epsilon_func):
        if epsilon_func is None:
            self._epsilon_func = epsilon_func
        elif type(epsilon_func) in [float, int]:
            self._epsilon_func = lambda x: epsilon_func
        elif isinstance(epsilon_func, lambda x: None):
            self._epsilon_func = epsilon_func
        else:
            raise ValueError("This value is not a valid epsilon schedule")

    @property
    def iteration(self):
        return self.trainer.iteration

    @iteration.setter
    def iteration(self, value):
        self.trainer.iteration = value

    @property
    def temperature(self):
        if self._temp_func is None:
            return 0
        return self._temp_func(self.iteration)

    @temperature.setter
    def temperature(self, temp_func):
        if temp_func is None:
            self._temp_func = temp_func
        elif type(temp_func) in [float, int]:
            self._temp_func = lambda x: temp_func
        elif isinstance(temp_func, lambda x: None):
            self._temp_func = temp_func
        else:
            raise ValueError("This value is not a valid temperature schedule")

    @property
    def network_size(self):
        total = 0
        for w in self.get_weights():
            total += w.size
        for b in self.get_biases():
            total += b.size
        return total


class QTrainer (Trainer):
    def default_params(self):
        return {
            "mode": "episodic",
            "learn_rate": 1e-4,
            "epochs": 1,
            "batch_size": 1,
            "replay_size": 20
        }

    def online(self, state, action, reward, learn_rate = None, **kwargs):
        """ Train over a single state, action, and reward """
        self.offline([state], [action], [reward], 1, 1, learn_rate, **kwargs)

    def offline(self, states = None, actions = None, rewards = None,
                batch_size = None, epochs = None, learn_rate = None):
        """
        Trains the agent over a sets of state, action, and reward

        Note:
            If states, actions, and rewards are None, full history used

        Args:
            states: (array[]) List of states
            actions: (array[]) Actions taken on states
            rewards: (float[]) Rewards for each action
            batch_size: (int) Number of samples in each minibatch
            epochs: (int) Number of iterations over the entire dataset
        """
        # Default to defaults if parameters not given
        if learn_rate is None:
            learn_rate = self.params["learn_rate"]
        if epochs is None:
            epochs = self.params["epochs"]
        if batch_size is None:
            batch_size = self.params["batch_size"]
        if states is None or actions is None or rewards is None:
            states = self.agent.states
            actions = self.agent.actions
            rewards = self.agent.rewards
        # Train for each epoch
        for epoch in range(epochs):
            # Calculate targets from states, actions, and rewards
            targets = [self.agent.target(state, action, reward)
                       for (state, action, reward) in
                       zip(states, actions, rewards)]
            # Separate into batches and train
            summary = self.batch(states, targets, batch_size, learn_rate)
        # Record if Tensorboard recording enabled and write summary to file
        if self.record and (self.iteration % self.tensorboard_interval == 0)\
                and summary is not None:
            self.writer.add_summary(summary, self.iteration)
        # Increase iteration counter
        self.iteration += 1

    def batch(self, states, targets, batch_size, learn_rate = None):
        """
        Trains the agent over a batch of states and targets

        Args:
            states: (array[]) List of states
            targets: (array[]) List of targets for each state
            batch_size: (int) Number of samples in each minibatch
            learn_rate: (float) Learning rate
        Returns:
            (tensorboard summary) Training summary for tensorboard recording
        """
        # Default learning rate if none provided
        if learn_rate is None:
            learn_rate = self.params["learn_rate"]
        # Batching
        # Shuffle all indicies
        order = list(range(len(states)))
        shuffle(order)
        # Chunk index list into batches of desired size
        batches = list(util.chunk(order, batch_size))
        # Train over each minibatch
        summary = None
        for batch in batches:
            summary = self.agent.update([states[b] for b in batch],
                                        [targets[b] for b in batch],
                                        learn_rate)
        return summary
