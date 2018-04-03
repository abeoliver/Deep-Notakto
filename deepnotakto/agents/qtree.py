#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: agents/qtree.py                                              #
#  Abraham Oliver, 2018                                               #
#######################################################################

# Import dependencies
from collections import deque
from copy import copy
from pickle import dump
from random import shuffle, sample

import numpy as np
import tensorflow as tf

import deepnotakto.util as util
from deepnotakto.agents.Q import Q
from deepnotakto.trainer import Trainer
from deepnotakto.treesearch import search, GuidedNode


class QTree (Q):
    """
    An agent using Silver et al's AlphaZero algorithm for training and eval
    
    Methods:
        clear - Reset the agent
        policy - Calculuate the policy for a given state
        get_Q - Get a Q-value matrix (a square policy vector)
        value - Calculate the policy for a given state
        raw - Get the raw output from the model after passing a given state
        update - Run a training update over given training data
        train - Use the trainer to train the model
        add_episode - Add a given state etc. tuple to the current episode
        save_episode - Save episode to the memory queue (trains if requested)
        save_point - Save a given state-policy-winner to the memory queue
        new_node - Create a compatibly-typed node with given parameters
        self_play - Run games of self play to generate training data
        act - Act on an evironment
        duplicative_dict - Get a dict that is sufficient to replicate the agent
    """
    def __init__(self, game_size, hidden_layers, guided = True, beta = None,
                 name = None, player_as_input = True, initialize = True,
                 classifier = None, iterations = 0, params = {},
                 max_queue = 100, play_simulations = 1000, mode = "q",
                 default_temp = 1, states = None, policies = None,
                 winners = None, guided_explore = 1,
                 activation_func = "identity", activation_type = "hidden",
                 **kwargs):
        """
        Initialize a QTree agent
        Args:
            game_size: (int) Side length of the boaed
            hidden_layers: (int[]) Number of neurons in each hidden layer
            classtpye: (Node Class) Class to use for new nodes
            beta: (float) L2 regularization hyperparameter
            name: (string) Agent name
            player_as_input: (bool) Pass the current player as part of input?
            initialize: (bool) Run model and tensorflow initializer?
            classifier: (string) Unique identifier for agent
            iterations: (int) Current iteration of model
            params: (dict) Training parameters
            max_queue: (int) Maximum length of the memory queue
            play_simulations: (int) Defult number of simulations per move
            mode: ("q" or "search") Method for move choosing
            default_temp: (float) Default temperature hyperparameter
            states: (array[]) Memory queue of states
            policies: (array[]) Memory queue of policies
            winners: (int[]) Memory queue of winners
            guided_explore: (float) Treesearch exploration hyperparameter
            activation_func: (string) Activaion function to be used on neurons
            activation_type: ("all" or "hidden") Neurons to apply activation
            kwargs: (dict) Arguments to pass to trainer
        """
        # Get classifier if none given
        if classifier is None:
            classifier = util.unique_classifier()
        # Get name if never given
        if name is None:
            name = "QTree({})".format(classifier)
        # Call parent initializer
        super(QTree, self).__init__(game_size, hidden_layers, 0,
                                    beta, name, False, classifier,
                                    iterations, params, max_queue, None, None,
                                    activation_func, activation_type)
        # Add value node
        self.layers[-1] += 1
        # Add player as input node
        self.player_as_input = player_as_input
        if player_as_input:
            self.layers[0] += 1
        # Run model initializer
        if initialize:
            self.initialize(params = params, iterations = iterations, **kwargs)

        # Like actions and rewards, record tree decided policies and the winners
        # If any one of the given sets is empty, set each to none
        # Note: self.states is set by parent initializers
        if type(None) in [type(states), type(policies), type(winners)]:
            self.policies = deque(maxlen = max_queue)
            self.winners = deque(maxlen = max_queue)
        else:
            self.states = deque(states, maxlen = max_queue)
            self.policies = deque(policies, maxlen = max_queue)
            self.winners = deque(winners, maxlen = max_queue)

        self.guided = guided
        self.play_simulations = play_simulations
        self._act_mode = mode
        self.default_temp = default_temp
        self.guided_explore = guided_explore
        self.node_defaults = self._node_defaults()

    def _init_trainer(self, params = None, **kwargs):
        """ Initialize a trainer object """
        self.trainer = QTreeTrainer(self, params = params, **kwargs)

    def _init_model(self, weights = None, biases = None):
        """
        Initialize the neural network with given or random weights and biases

        Args:
            weights: (array[]) Initial weights
            biases: (arrayp[]) Initial biases
        """
        # Parent initializer
        super(QTree, self)._init_model(weights, biases)
        # Policy head
        self._probabilities = tf.nn.softmax(self.y[:, 1:])
        # Value head
        self._value = tf.tanh(self.y[:, 0])

    def _init_training_vars(self):
        """ Initialize computational graph nodes for training procedure """
        with self._graph.as_default():
            # Target probabilities
            self.prob_targets = tf.placeholder(
                tf.float32, shape = [None, self.layers[-1] - 1],
                name = "probability_targets")
            # Target values
            self.winner_targets = tf.placeholder(tf.float32, shape = [1, None],
                                                 name = "winner_target")
            # SGD learning rate
            self.learn_rate = tf.placeholder(tf.float32)
            # SGD loss
            self._loss = self._get_loss_function()
            # SGD Optimizer
            self._optimizer = tf.train.GradientDescentOptimizer(
                learning_rate = self.learn_rate, name = "optimizer")
            # Get gradients for clipping
            self._gradients = self._optimizer.compute_gradients(self._loss)
            # Verify finite and real
            name = "FiniteGradientVerify"
            grads = [(tf.verify_tensor_all_finite(
                g, msg = "Inf or NaN Gradients for {}".format(v.name),
                name = name), v)
                     for g, v in self._gradients]
            # Clip gradients according to threshold
            self._clipping_threshold = tf.placeholder(tf.float32,
                                                      name="grad_clip_thresh")
            self._clipped_gradients = [
                (tf.clip_by_norm(g, self._clipping_threshold), v)
                for g, v in grads]
            # Update operation (minimizer)
            self.update_op = self._optimizer.apply_gradients(
                self._clipped_gradients, name = "update")
            # Tensorboard saving operation
            self.summary_op = tf.summary.merge_all()

    def _get_loss_function(self):
        """
        Design and produce the loss function for gradient descent

        Returns:
            func: array[] -> float
        """
        with tf.name_scope("loss"):
            # Winner / Value loss
            val_loss = tf.reduce_sum(
                tf.square(self.winner_targets - self._value),
                name = "value_loss")
            tf.summary.scalar("Value_loss", val_loss)
            # Cross entropy for policy
            prob_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                labels = self.prob_targets, logits = self.y[:, 1:],
                name = "policy_loss"))
            tf.summary.scalar("Policy_loss", prob_loss)
            # L2 Regularization
            # Initialize to zero
            self.l2 = 0.0
            # Regularization (if beta was set by user)
            if self.beta is not None:
                with tf.name_scope("regularization"):
                    self.l2 = self._l2_recurse(self.w)
                    tf.summary.scalar("L2", self.l2)
                    beta = tf.constant(self.beta)
                    # Full loss function
                loss = tf.add(val_loss + prob_loss, beta * tf.square(self.l2),
                              name = "loss_with_regularization")
            else:
                # Full loss function
                loss = tf.add(val_loss, prob_loss, name = "loss")
            # Vefify real and finite
            loss = tf.verify_tensor_all_finite(loss, name = "FiniteVerify",
                                               msg = "Inf or NaN loss")
            tf.summary.scalar("Loss", loss)
            return loss

    def clear(self):
        """ Reset agent properties """
        super(QTree, self).clear()
        # Reset qtree specific properties
        self.policies = deque(maxlen = self.max_queue)
        self.winners = deque(maxlen = self.max_queue)

    def policy(self, state):
        """
        Pass a state to the model and get the policy head output

        Args:
            state: (array or array[]) State or states to feed to network
        Returns:
            (array or array[]) Policy or policies for given state(s)
        """
        probs = self._probabilities.eval(
            session = self.session,
            feed_dict = {self.x: self.get_passable_state(state)})
        # If only one state passed in, return single instead of one-element list
        if probs.size == self.layers[-1] - 1:
            return probs[0]
        return probs

    def get_Q(self, state):
        """
        Get a policy and return a reshaped version

        Note: Needed for the general Q-based-agent API

        Args:
            state: (array or array[]) State or states to feed to network
        """
        return self.policy(state).reshape(state.shape)

    def value(self, state):
        """
        Pass a state into the model and take value head output

        Args:
            state: (array or array[]) State or states to feed to network
        Returns:
            (float or float[]) Value(s) for given state(s)
        """
        val = self._value.eval(
            session = self.session,
            feed_dict = {self.x: self.get_passable_state(state)})
        # If only one state passed in, return single instead of one-element list
        if val.size == 1:
            return val[0]
        return val

    def raw(self, state):
        """
        Pass a state into the model and get the raw output from the network

        Args:
            state: (array or array[]) State or states to feed to network
        Returns:
            (array or array[]) Output(s) for given state(s)
        """
        out = self.y.eval(session = self.session,
                          feed_dict = {self.x: self.get_passable_state(state)})
        # If only one state passed in, return single instead of one-element list
        if out.size == 1:
            return out[0]
        return out

    def update(self, states, policies, winners, learn_rate = .01):
        """
        Update the model with a given set of data

        Args:
            states: (array[]) States to train over
            policies: (array[]) Policies to train over
            winners: (int[]) Winners to train over
            learn_rate: (float) Learning rate for SGD
        Returns:
            Tensorboard Summary
        """
        # Clean winner input
        if type(winners) == list:
            winners_ = np.array([winners])
        else:
            winners_ = np.array(winners)
        # Flatten states and probs
        states_ = np.array(self.get_passable_state(states), dtype = np.float32)
        policies_ = np.array([np.reshape(p, -1) for p in policies],
                             dtype = np.float32)
        # Construct feed dictionary for the optimization step
        feed_dict = {self.x: states_, self.prob_targets: policies_,
                     self.winner_targets: winners_, self.learn_rate: learn_rate,
                     self._clipping_threshold: self.clip_thresh}
        # Optimize the network and return the tensorboard summary information
        return self.session.run([self.summary_op, self.update_op],
                                feed_dict = feed_dict)[0]

    def train(self, **kwargs):
        """
        Run a training process with the trainer

        Args:
            kwargs: (dict) Arguments for trainer
        """
        self.trainer.train(**kwargs)

    def add_episode(self, state, policy, value = 0):
        """
        Add a state-policy-value data point to the current episode

        Args:
            state: (array) Game state
            policy: (array) Policy used
            value: (float) Value or winner
        """
        self.episode.append((state, policy, value))

    def save_episode(self):
        """ Add elements in the episode to memory """
        # If episode hasn't been used, do nothing
        if len(self.episode) == 0:
            return None
        # Get winner
        winner = 1 if self.episode[-1][2] == 0 else -1
        # Loop through all timesteps and add to memory with the correct winner
        # Winner is assumed to be value of the final datapoint in the episode
        for s, p, _ in self.episode:
            self.save_point(s, p, winner)
        # Train if requested
        if self.params["train_live"]:
            self.train()
        # Clear episode
        self.new_episode()

    def save_point(self, state, probs, winner):
        """
        Save a data point to the memory queue

        Note: use for self-play generated data points

        Args:
            state: (array) Board state
            probs: (array) Probability distribution
            winner: (int) Winner of game
        """
        self.states.append(state)
        self.policies.append(probs)
        self.winners.append(winner)

    def new_node(self, **kwargs):
        """ Create a new node with given parameters resolving to defaults """
        params = self._node_param_resolve(**kwargs)
        return GuidedNode(**params)

    def self_play(self, games, simulations, save_every = 0,
                  save_name = None, train = True, guided = None):
        """
        Runs games of self play for training data generation
        
        Args:
            games: (int) Number of games to play
            simulations: (int) Number of simulations to run per move
            save_every: (int) Number of games to save after
            save_name: (string) Filename to save agent under
            train: (bool) Train after every game?
        """
        # Clean input
        save_every = int(save_every)
        games = int(games)
        simulations = int(simulations)
        if save_every > 0 and save_name is None:
            save_name = self.name + ".npz"
        if guided is None:
            guided = self.guided
        # Run self play
        for game in range(games):
            states = []
            policies = []
            # Start with a root node
            node = self.new_node(network = self)
            while True:
                # Separate node from tree and reset it
                node.separate()
                # Run a guided search
                search(node, simulations, modified = guided)
                # Save the information from this node
                states.append(node.state)
                policy = node.get_policy(self.default_temp)
                policies.append(policy.reshape(node.state.shape))
                # Choose move based on policy
                node = node.choose_by_policy(policy)
                # If terminal, backpropogate winner and save node data
                if node.winner != 0:
                    winner = node.winner
                    states.append(node.state)
                    policies.append(np.zeros(node.state.shape))
                    break
                elif not node.action_space():
                    # Player that made this position wins
                    winner = node.player
                    states.append(node.state)
                    policies.append(np.ones(node.state.shape) / node.state.size)
                    break
            # Add these data points
            for i in range(len(states)):
                current_player = 1 + (i % 2)
                self.save_point(states[i], policies[i],
                                1 if winner == current_player else -1)
            # Train
            if train:
                self.train()
            # Save model
            if save_every > 0 and game % save_every == 0:
                self.save(save_name)

    def act(self, env, mode = None):
        """
        Act on an environment using the desired mode

        Args:
            env: (Environment) Game environment to play on
            mode: ("q" or "search") Desired method for choosing an action
        Returns:
            (dict) Observation from environment
        """
        if mode is None:
            mode = self.mode
        # Select a mode to use and get move
        return {"q": self._q_act, "search": self._search_act}[mode](env)

    def _q_act(self, env):
        """
        Choose a move to play deterministically and purely  based on a policy

        Args:
            env: (Environment) Environement to play on
        Return:
            (dict) Observation from environment
        """
        return env.act(self.get_action(env.observe()))

    def _search_act(self, env):
        """
        Choose a move to play by the result of a tree search

        Args:
            env: (Environment) Environement to play on
        Returns:
            (dict) Observation from environment
        """
        # Start with a root node
        state = env.observe()
        node = self.new_node(state = state, network = self)
        if not node.action_space():
            # Run a guided search
            search(node, self.play_simulations, modified = True)
            # Save to memory
            policy = node.get_policy(self.default_temp).reshape(state.shape)
            self.add_episode(state, policy)
            # Choose move based on policy
            node = node.choose_by_visits()
        else:
            node = node.random_move(False)
            self.add_episode(state, np.ones(state.shape) / state.size, -1)
        # Get the desired move
        move = np.zeros(node.state.shape, dtype = np.int8)
        move[node.edge // move.shape[0], node.edge % move.shape[0]] = 1
        # Play the move and return the result
        return env.act(move)

    @property
    def mode(self):
        return self._act_mode

    @mode.setter
    def mode(self, new):
        if new.lower() in ["q", "search"]:
            self._act_mode = new.lower()

    def duplicative_dict(self):
        """ Get dictionary containing information needed to replicate agent """
        return {"game_size": self.size, "hidden_layers": self.layers[1:-1],
                "weights": self.get_weights(), "biases": self.get_biases(),
                "name": self.name,
                "beta": self.beta, "classifier": self.classifier,
                "params": self.params, "iterations": self.iteration,
                "max_queue": self.max_queue,
                "play_simulations": self.play_simulations, "mode": self.mode,
                "default_temp": self.default_temp, "states": self.states,
                "policies": self.policies, "winners": self.winners,
                "player_as_input": self.player_as_input,
                "guided_explore": self.guided_explore,
                "activation_func": self.activation_func_name,
                "activation_type": self.activation_type,
                "tensorboard_interval": self.trainer.tensorboard_interval,
                "tensorboard_path": self.trainer.tensorboard_path}

    def _node_defaults(self):
        """ Get the defaults for a new node """
        return {
            "state": None,
            "parent": None,
            "edge": None,
            "visits": 0,
            "remove_unvisited_losses": True,
            "total_value": 0,
            "prior": 0,
            "explore": 1
        }

    def _node_param_resolve(self, **kwargs):
        """ If a parameter is not given, retrieve from defaults """
        params = kwargs
        if params is None:
            return self.node_defaults
        else:
            for key in self.node_defaults:
                if key not in params:
                    params[key] = self.node_defaults[key]
            return params


class QTreeTrainer (Trainer):
    def default_params(self):
        return {
            "learn_rate": 1e-4,
            "epochs": 1,
            "batch_size": 1,
            "replay_size": 20,
            "train_live": False
        }

    def train(self, **kwargs):
        """ Train over entire memory queue """
        self.train_on_set(self.agent.states,
                          self.agent.policies,
                          self.agent.winners)

    def train_on_set(self, states, policies, winners, **kwargs):
        """ Train over a given set of states, policies, and winners """
        # Randomly sample indecies from memory (aka memory replay)
        size = len(self.agent.states)
        if size > self.params["replay_size"]:
            indexes = sample(range(size), self.params["replay_size"])
        else:
            indexes = range(size)
        # Train on this sample
        self.offline([states[i] for i in indexes],
                     [policies[i] for i in indexes],
                     [winners[i] for i in indexes],
                     **kwargs)

    def offline(self, states = None, policies = None, winners = None,
                batch_size = None, epochs = None, learn_rate = None):
        """ Train over a dataset """
        if learn_rate is None:
            learn_rate = self.params["learn_rate"]
        if epochs is None:
            epochs = self.params["epochs"]
        if batch_size is None:
            batch_size = self.params["batch_size"]
        if states is None or policies is None or winners is None:
            states = self.agent.states
            policies = self.agent.policies
            winners = self.agent.winners
        # Train for each epoch
        for epoch in range(epochs):
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
                summary = self.agent.update(bs, bp, bw, learn_rate)
            # Record if Tensorboard recording enabled
            if self.record and\
                    (self.iteration % self.tensorboard_interval == 0)\
                    and summary is not None:
                # Write summary to file
                self.writer.add_summary(summary, self.iteration)
        # Increase iteration counter
        self.iteration += 1

    def online(self, state, probs, winner, learn_rate = None,
               epochs = 1, **kwargs):
        """ Gets a callable function for online params """
        self.offline(states = [state], policies = [probs], winners = [winner],
                     batch_size = 1, epochs = epochs, learn_rate = learn_rate)
