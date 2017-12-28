# qtree.py
# Abraham Oliver, 2017
# DeepNotakto Project

import numpy as np
import tensorflow as tf
from collections import deque

import deepnotakto.util as util
from deepnotakto.agents.Q import Q
from deepnotakto.trainer import Trainer

class QTree (Q):
    def __init__(self, layers, gamma = .8, beta = None, name = None,
                 initialize = True, classifier = None, iterations = 0,
                 params = {"mode": "tree"}, max_queue = 100, **kwargs):
        # Get classifier
        if classifier == None:
            classifier = util.unique_classifier()
        # Get name
        if name == None:
            name = "QTree({})".format(classifier)
        # Add value node
        layers[-1] = layers[-1] + 1
        # Call parent initializer
        super(QTree, self).__init__(layers, gamma, beta, name, initialize, classifier,
                                    iterations, params, max_queue, None, None, **kwargs)
        # Like states and actions, record tree decided probabilities
        self.probabilities = deque(maxlen = max_queue)
        self.winners = deque(maxlen = max_queue)

    def initialize(self, weights = None, biases = None, force = False,
                   params = None, **kwargs):
        """
        Initialize the model
        Parameters:
            weights (List of (N, M) arrays with variable size) - Initial weights
            biases (List of (1, N) arrays with variable size) - Initial biases
            force (bool) - Initialize, even if already initialized
            params (dict) - Training parameters
            KWARGS passed to trainer initializer
        """
        if not self.initialized or force:
            # Create a tensorflow session for all processes to run in
            self._graph = tf.Graph()
            self.session = tf.Session(graph = self._graph)
            # Initialize model
            self.init_model(weights = weights, biases = biases)
            # Initialize params variables like the loss and the optimizer
            self._init_training_vars()
            # Initialize trainer (passing the agent as a parameter)
            self.trainer = QTreeTrainer(self, params = params, **kwargs)
            self.initialized = True

    def init_model(self, weights = None, biases = None):
        super(QTree, self).init_model(weights, biases)
        self._probabilities = tf.nn.softmax(self.y[:, 1:])
        self._value = tf.sigmoid(self.y[:, 0])

    def _init_training_vars(self):
        """Initialize params procedure"""
        with self._graph.as_default():
            with tf.name_scope("params"):
                # Targets
                # Probabilities
                self.prob_targets = tf.placeholder(tf.float32,
                                                   shape = [None, self.layers[-1] - 1],
                                                   name = "probability_targets")
                # Winner
                self.winner_targets = tf.placeholder(tf.float32, shape = [None, 1],
                                                     name = "winner_target")
                # Learning rate
                self.learn_rate = tf.placeholder(tf.float32)
                # Loss
                self.loss = self._get_loss_function()
                # Optimizer
                self._optimizer = tf.train.GradientDescentOptimizer(learning_rate = self.learn_rate,
                                                                    name = "optimizer")
                # Updater (minimizer)
                self.update_op = self._optimizer.minimize(self.loss, name ="update")
                # Tensorboard
                self.summary_op = tf.summary.merge_all()

    def _get_loss_function(self):
        # Winner / Value loss
        val_loss = tf.reduce_sum(tf.square(self.winner_targets - self._value),
                                 name = "value_loss")
        tf.summary.scalar("Value_loss", val_loss)
        # Cross entropy for policy
        # prob_loss = tf.reduce_sum(tf.matmul(tf.transpose(self.prob_targets),
        #                                      tf.log(self._probabilities)))
        prob_loss = tf.nn.softmax_cross_entropy_with_logits(labels = self.prob_targets,
                                                            logits = self.y[:, 1:],
                                                            name = "policy_loss")
        # L2 Regularization
        self.l2 = 0.0
        # Loss and Regularization
        if self.beta != None:
            with tf.name_scope("regularization"):
                self.l2 = self._l2_recurse(self.w)
                tf.summary.scalar("L2", self.l2)
                beta = tf.constant(self.beta)
        else:
            beta = tf.constant(0.0)
        # Full loss (negative prob loss is built into the cross entropy function)
        loss = tf.reduce_sum(val_loss + prob_loss +
                             beta * tf.square(self.l2),
                             name = "loss")
        loss = tf.verify_tensor_all_finite(loss, name = "FiniteVerify",
                                           msg = "Inf or NaN values")
        tf.summary.scalar("Loss", loss)
        return loss

    def clear(self):
        super(QTree, self).clear()
        self.probabilities = deque(maxlen = self.max_queue)
        self.winners = deque(maxlen=self.max_queue)

    def act(self, env):
        # Get current environment state
        state = env.observe()
        # Get desired action
        action = self.get_action(state)
        # Apply the action and retrieve observation
        observation = env.act(action)
        # Train online (may be avoided within the function w/ params params)
        if self.params["mode"] == "online":
            self.train("online")
        # Return the results
        return observation

    def get_probs(self, state):
        probs = self._probabilities.eval(session = self.session,
                                         feed_dict = {self.x: [np.reshape(state, -1)]})
        if probs.size == self.layers[-1] - 1:
            return probs[0]
        return probs

    def get_Q(self, state):
        return self.get_probs(state)

    def get_winner(self, state):
        if type(state) == list:
            feed = [np.reshape(s, -1) for s in state]
        elif type(state) == np.ndarray:
            feed = [np.reshape(state, -1)]
        winner = self._value.eval(session = self.session,
                                  feed_dict = {self.x: feed})
        if winner.size == 1:
            return winner[0]
        return winner

    def update(self, states, probs, winners, learn_rate = .01):
        # Clean winner input
        if type(winners) == list:
            WINNERS = np.array([winners])
        else:
            WINNERS = np.array(winners)
        # Flatten states and probs
        STATES = np.array([np.reshape(s, -1) for s in states], dtype = np.float32)
        PROBS = np.array([np.reshape(p, -1) for p in probs], dtype = np.float32)
        # Construct feed dictionary for the optimization step
        feed_dict = {self.x: STATES, self.prob_targets: PROBS,
                     self.winner_targets: WINNERS,
                     self.learn_rate: learn_rate}
        # Optimize the network and return the tensorboard summary information
        #print(self.session.run(self.loss, feed_dict = feed_dict))
        return self.session.run([self.summary_op, self.update_op], feed_dict = feed_dict)[0]

    def save(self, name):
        # Todo
        pass

    def add_episode(self, *args, **kwargs):
        # Not necessary, removing functionality
        pass

    def save_episode(self, *args, **kwargs):
        # Not necessary, removing functionality
        pass

class QTreeTrainer (Trainer):
    def default_params(self):
        return {
            "learn_rate": 1e-4,
            "rotate": False,
            "epochs": 1,
            "batch_size": 1,
            "replay_size": 20
        }

    def offline(self, states = None, probs = None, winners = None, batch_size = None,
                epochs = None, learn_rate = None, rotate = None):
        if learn_rate == None:
            learn_rate = self.params["learn_rate"]
        if rotate == None:
            rotate = self.params["rotate"]
        if epochs == None:
            epochs = self.params["epochs"]
        if batch_size == None:
            batch_size = self.params["batch_size"]
        if states == None or probs == None or winners == None:
            states = self.agent.states
            probs = self.agent.probabilities
            winners = self.agent.winners
        # Train for each epoch
        for epoch in range(epochs):
            # Rotate if required
            if rotate:
                states, probs = self.get_rotations(states, probs)
            # Separate into batches and train
            # Batching
            # Shuffle all indicies
            order = list(range(len(states)))
            shuffle(order)
            # Chunk index list into batches of desired size
            batches = list(self.chunk(order, batch_size))
            summary = None
            for batch in batches:
                # Get the states and targets for the indicies in the batch and update
                summary = self.agent.update([states[b] for b in batch],
                                            [probs[b] for b in batch],
                                            [winners[b] for b in batch],
                                            learn_rate)
            # Record if Tensorboard recording enabled
            if self.record and (self.iteration % self.tensorboard_interval == 0) \
                    and summary != None:
                # Write summary to file
                self.writer.add_summary(summary, self.iteration)
            # Increase iteration counter
            self.iteration += 1