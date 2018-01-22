# qtree.py
# Abraham Oliver, 2017
# DeepNotakto Project

from collections import deque
from copy import copy
from pickle import dump
from random import shuffle, sample

import numpy as np
import tensorflow as tf

import deepnotakto.util as util
from deepnotakto.agents.Q import Q
from deepnotakto.trainer import Trainer
from deepnotakto.treesearch import GuidedNotaktoNode, search


class QTree (Q):
    def __init__(self, layers, gamma = .8, beta = None, name = None,
                 initialize = True, classifier = None, iterations = 0,
                 params = {}, max_queue = 100, play_simulations = 10,
                 act_mode = "search", default_temp = 1, states = None,
                 policies = None, winners = None, guided_explore = 1, **kwargs):
        # Get classifier
        if classifier == None:
            classifier = util.unique_classifier()
        # Get name
        if name == None:
            name = "QTree({})".format(classifier)
        # Add value node if not added yet
        layers = copy(layers)
        if layers[0] == layers[-1]:
            layers[-1] += 1
        # Call parent initializer
        super(QTree, self).__init__(layers, gamma, beta, name, initialize, classifier,
                                    iterations, params, max_queue, None, None, **kwargs)
        # Like states and actions, record tree decided policies
        if type(states) != type(None):
            self.states = deque(states, maxlen = max_queue)
        if type(policies) != type(None):
            self.policies = deque(policies, maxlen = max_queue)
        else:
            self.policies = deque(maxlen = max_queue)
        if type(winners) != type(None):
            self.winners = deque(winners, maxlen = max_queue)
        else:
            self.winners = deque(maxlen = max_queue)
        # Number of simulations to run on each move when playing
        self.play_simulations = play_simulations
        # Mode of acting to do
        self.act_mode = act_mode
        # Default policy temperature
        self.default_temp = default_temp
        # Exploration constant for guided search
        self.guided_explore = guided_explore

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
        self._value = tf.tanh(self.y[:, 0])

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
                self.winner_targets = tf.placeholder(tf.float32, shape = [1, None],
                                                     name = "winner_target")
                # Learning rate
                self.learn_rate = tf.placeholder(tf.float32)
                # Loss
                self._loss = self._get_loss_function()
                # Optimizer
                self._optimizer = tf.train.GradientDescentOptimizer(learning_rate =
                                                                    self.learn_rate,
                                                                    name = "optimizer")
                # Get gradients
                self._gradients = self._optimizer.compute_gradients(self._loss)
                # Verify finite and real
                name = "FiniteGradientVerify"
                grads = [(tf.verify_tensor_all_finite(g, msg = "Inf or NaN Gradients for {}".format(v.name),
                                                      name = name), v)
                         for g, v in self._gradients]
                # Clip gradients
                self._clipping_threshold = tf.placeholder(tf.float32, name="grad_clip_thresh")
                self._clipped_gradients = [(tf.clip_by_norm(g, self._clipping_threshold), v)
                                           for g, v in grads]
                # Updater (minimizer)
                self.update_op = self._optimizer.apply_gradients(self._clipped_gradients,
                                                                 name = "update")
                # Tensorboard
                self.summary_op = tf.summary.merge_all()

    def _get_loss_function(self):
        # Winner / Value loss
        val_loss = tf.reduce_sum(tf.square(self.winner_targets - self._value), name = "value_loss")
        tf.summary.scalar("Value_loss", val_loss)
        # Cross entropy for policy
        prob_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = self.prob_targets,
                                                                          logits = self.y[:, 1:],
                                                                          name = "policy_loss"))
        tf.summary.scalar("Policy_loss", prob_loss)
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
        loss = tf.add(val_loss + prob_loss, beta * tf.square(self.l2), name = "loss")
        loss = tf.verify_tensor_all_finite(loss, name = "FiniteVerify",
                                           msg = "Inf or NaN loss")
        tf.summary.scalar("Loss", loss)
        return loss

    def clear(self):
        super(QTree, self).clear()
        self.policies = deque(maxlen = self.max_queue)
        self.winners = deque(maxlen = self.max_queue)

    def policy(self, state):
        probs = self._probabilities.eval(session = self.session,
                                         feed_dict = {self.x: [np.reshape(state, -1)]})
        if probs.size == self.layers[-1] - 1:
            return probs[0]
        return probs

    def get_Q(self, state):
        return self.policy(state).reshape(state.shape)

    def value(self, state):
        if type(state) == list:
            feed = [np.reshape(s, -1) for s in state]
        elif type(state) == np.ndarray:
            feed = [np.reshape(state, -1)]
        val = self._value.eval(session = self.session,
                               feed_dict = {self.x: feed})
        if val.size == 1:
            return val[0]
        return val

    def raw(self, state):
        if type(state) == list:
            feed = [np.reshape(s, -1) for s in state]
        elif type(state) == np.ndarray:
            feed = [np.reshape(state, -1)]
        out = self.y.eval(session = self.session,
                          feed_dict = {self.x: feed})
        if out.size == 1:
            return out[0]
        return out

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
                     self.winner_targets: WINNERS, self.learn_rate: learn_rate,
                     self._clipping_threshold: self.clip_thresh}
        # Optimize the network and return the tensorboard summary information
        #print(self.session.run(self.loss, feed_dict = feed_dict))
        return self.session.run([self.summary_op, self.update_op], feed_dict = feed_dict)[0]

    def save_as_prob_model(self, name):
        layers = copy(self.layers)
        layers[-1] -= 1
        with open(name, "wb") as outFile:
            dump({"weights": [w.eval(session = self.session)[:, 1:] for w in self.w],
                  "biases": [b.eval(session = self.session)[:, 1:] for b in self.b],
                  "layers": layers, "gamma": self.gamma, "name": self.name,
                  "beta": self.beta, "classifier": self.classifier,
                  "params": self.params, "iterations": self.trainer.iteration},
                 outFile)

    def train(self, **kwargs):
        self.trainer.train(**kwargs)

    def add_episode(self, state, policy, value = 0):
        self.episode.append((state, policy, value))

    def save_episode(self):
        # If episode hasn't been used, do nothing
        if len(self.episode) == 0:
            return None
        # Get winner
        winner = 1 if self.episode[-1][2] == 0 else -1
        # Loop through all time steps and add them to memory with the correct winner
        for s, p, _ in self.episode:
            self.states.append(s)
            self.policies.append(p)
            self.winners.append(winner)
        # Train if requested
        if self.params["train_live"]:
            self.train()
        # Clear episode
        self.new_episode()

    def save_point(self, state, probs, winner):
        self.states.append(state)
        self.policies.append(probs)
        self.winners.append(winner)

    def self_play(self, games, simulations, save_every = 0, save_name = None, train = True):
        # Clean input
        save_every = int(save_every)
        games = int(games)
        simulations = int(simulations)
        if save_every > 0 and save_name == None:
            save_name = self.name + ".npz"
        # Run self play
        for game in range(games):
            states = []
            policies = []
            # Start with a root node
            node = GuidedNotaktoNode(np.zeros(self.shape), self, explore = self.guided_explore,
                                     remove_unvisited_losses = False)
            while True:
                # Separate node from tree and reset it
                node.separate()
                # Run a guided search
                search(node, simulations, guided = True)
                # Save the information from this node
                states.append(node.state)
                policy = node.get_policy(self.default_temp)
                policies.append(policy.reshape(node.state.shape))
                # Choose move based on policy
                node = node.choose_by_policy(policy)
                # If terminal, backpropogate winner and save (state, policy, winner)
                if node.winner != 0:
                    winner = node.winner
                    states.append(node.state)
                    policies.append(np.zeros(node.state.shape))
                    break
                elif node.action_space() == []:
                    # Player that made this position wins
                    winner = node.player
                    states.append(node.state)
                    policies.append(np.ones(node.state.shape) / node.state.size)
                    break
            # Add these data points
            for i in range(len(states)):
                current_player = 1 + (i % 2)
                self.save_point(states[i], policies[i], 1 if winner == current_player else -1)
            # Train
            if train:
                self.train()
            # Save model
            if save_every > 0 and game % save_every == 0:
                self.save(save_name)

    def act(self, env, mode = None):
        if mode == None:
            mode = self.act_mode
        return {"q": self.q_act, "search" : self.search_act}[mode](env)

    def q_act(self, env):
        """ Gets a move for a given environment, plays it, and returns the result"""
        return env.act(self.get_action(env.observe()))

    def search_act(self, env):
        # Start with a root node
        state = env.observe()
        node = GuidedNotaktoNode(state, self)
        if node.action_space() != []:
            # Run a guided search
            search(node, self.play_simulations, guided = False)
            # Save to memory
            self.add_episode(state, node.get_policy(self.default_temp).reshape(state.shape))
            # Choose move based on policy
            node = node.choose_by_visits()
        else:
            node = node.random_move(False, False)
            self.add_episode(state, np.ones(state.shape) / state.size, -1)
        # Get the desired move
        move = np.zeros(node.state.shape, dtype = np.int8)
        move[node.edge // move.shape[0], node.edge % move.shape[0]] = 1
        # Play the move and return the result
        return env.act(move)

    def mode(self, new):
        if new.lower() in ["q", "search"]:
            self.act_mode = new.lower()

    def value_head(self):
        """ Returns the final weights responsible for the value head (most useful when no hidden layers """
        return self.get_weights(-1)[:, 0]

    def save(self, name):
        """
        Save the models parameters in a .npz file
        Parameters:
            name (string) - File name for save file
        """
        # Remove epsilon function from the parameters for pickle
        with open(name, "wb") as outFile:
            dump({"weights": self.get_weights(), "biases": self.get_biases(),
                  "layers": self.layers, "gamma": self.gamma, "name": self.name,
                  "beta": self.beta, "classifier": self.classifier, "params": self.params,
                  "iterations": self.iteration, "max_queue": self.max_queue,
                  "play_simulations": self.play_simulations, "act_mode": self.act_mode,
                  "default_temp": self.default_temp, "states": self.states,
                  "policies": self.policies, "winners": self.winners,
                  "tensorboard_interval": self.trainer.tensorboard_interval,
                  "tensorboard_path": self.trainer.tensorboard_path},
                 outFile)

class QTreeTrainer (Trainer):
    def default_params(self):
        return {
            "learn_rate": 1e-4,
            "rotate": False,
            "epochs": 1,
            "batch_size": 1,
            "replay_size": 20,
            "rotate_live": False,
            "train_live": False
        }

    def train(self, **kwargs):
        # Training loop: select memory replay and train
        for _ in range(self.params["epochs"]):
            # Randomly sample from memory
            size = len(self.agent.states)
            if size > self.params["replay_size"]:
                indexes = sample(range(size), self.params["replay_size"])
            else:
                indexes = range(size)
            # Train on this sample
            self.offline([self.agent.states[i] for i in indexes],
                         [self.agent.policies[i] for i in indexes],
                         [self.agent.winners[i] for i in indexes],
                         epochs = 1,  **kwargs)

    def offline(self, states = None, policies = None, winners = None, batch_size = None,
                epochs = None, learn_rate = None, rotate = None, rotate_live = None):
        if learn_rate == None:
            learn_rate = self.params["learn_rate"]
        if rotate == None:
            rotate = self.params["rotate"]
        if epochs == None:
            epochs = self.params["epochs"]
        if batch_size == None:
            batch_size = self.params["batch_size"]
        if rotate_live == None:
            rotate_live = self.params["rotate_live"]
        if states == None or policies == None or winners == None:
            states = self.agent.states
            policies = self.agent.policies
            winners = self.agent.winners
        # Train for each epoch
        for epoch in range(epochs):
            # Rotate if required
            if rotate and not rotate_live:
                states, policies, winners = self.get_rotations(states, policies, winners)
            # Separate into batches and train
            # Batching
            # Shuffle all indicies
            order = list(range(len(states)))
            shuffle(order)
            # Chunk index list into batches of desired size
            batches = list(self.chunk(order, batch_size))
            summary = None
            for batch in batches:
                bs, bp, bw = [[states[b] for b in batch],
                              [policies[b] for b in batch],
                              [winners[b] for b in batch]]
                if rotate_live:
                    bs, bp, bw = self.get_rotations(bs, bp, bw)
                # Get the states and targets for the indicies in the batch and update
                summary = self.agent.update(bs, bp, bw, learn_rate)
            # Record if Tensorboard recording enabled
            if self.record and (self.iteration % self.tensorboard_interval == 0) \
                    and summary != None:
                # Write summary to file
                self.writer.add_summary(summary, self.iteration)
        # Increase iteration counter
        self.iteration += 1

    def online(self, state, probs, winner, learn_rate = None, epochs = 1, **kwargs):
        """Gets a callable function for online params"""
        self.offline(states = [state], policies = [probs], winners = [winner],
                     batch_size = 1, epochs = epochs, learn_rate =learn_rate, **kwargs)

    def get_rotations(self, states, policies, winners):
        """Train over the rotated versions of each state and target (or probs)"""
        # Copy states so as not to edit outside of scope
        states = list(states)
        policies = list(policies)
        # Collect rotated versions of each state and target
        new_states = []
        new_policies = []
        new_winners = []
        for s, p, w in zip(states, policies, winners):
            # Aliases for rotating and renaming
            S = s
            P = p
            for i in range(3):
                # Rotate them
                rs = util.rotate(S)
                rp = util.rotate(P)
                # Add them to the new lists
                new_states.append(rs)
                new_policies.append(rp)
                new_winners.append(w)
                # Rename the rotations to be normal
                S = rs
                P = rp
        # Combine lists
        return [states + new_states, policies + new_policies, winners + new_winners]