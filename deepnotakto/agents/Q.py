# Q.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from random import choice
from datetime import datetime
from agent import Agent

class Q (Agent):
    def __init__(self, size, load_file_name = None, gamma = .8, trainable = True,
                epsilon = 0.0):
        """
        Initializes an Q learning agent
        Parameters:
            size (int) - Board side length
            load_file_name (string) - Path to load saved model from
            gamma (float [0, 1]) - Q-Learning hyperparameter (not used if model is loaded)
            trainable (bool) - Is the model trainable or frozen
            epsilon (float [0, 1]) - Epsilon for e-greedy exploration (only when training)
        Note:
            Initializes randomly if no model is given
        """
        # Call parent initializer
        super(QNoHidden, self).__init__()
        self.size = size
        self.targets = []
        self.trainable = trainable
        self.gamma = gamma
        self.epsilon = epsilon
        # Create a tensorflow session for all processes to run in
        self.session = tf.Session()
        # Load model if a file name is given
        if load_file_name != None:
            self.load(load_file_name)
        # Otherwise randomly initialize
        else:
            self.init_model()
        # Initialize training variables like the loss and the optimizer
        self.init_training_vars()
        
    def act(self, env, training_iteration = None, **kwargs):
        """
        Choose action, apply action to environment, and recieve reward
        Parameters:
            env (environment.Env) - Environment of the agent
            training_iteration (int) - If model is training, which iteration for e-greedy
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
        # Get Q values after move is applied
        newQ = self.get_Q(new_state)
        # Max new Q value
        maxQ = np.max(newQ)
        # Update Q for target
        Q[action_index] = reward + self.gamma * maxQ
        
        # Train network
        self.train([current_state], [Q])
        
        # Record state, action, reward, and target
        self.states.append(current_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.targets.append(Q)
        
        # Change epsilon if training
        if training_iteration != None:
            self.change_epsilon(training_iteration)
    
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
        """
        # Pass the state to the model and get array of Q-values
        return self.y.eval(session = self.session, 
                           feed_dict = {self.x: [self.flatten(state)]})[0]

    def init_model(self, w = None, b = None):
        """Randomly intitialize model"""
        with tf.name_scope("model"):
            s = self.size * self.size
            self.x = tf.placeholder(tf.float32, [None, s], name = "input")
            if type(w) != np.ndarray:
                self.w = tf.Variable(tf.random_normal([s, s]),
                                     trainable = self.trainable,
                                     name = "weights")
            else:
                self.w = tf.Variable(w, trainable = self.trainable, name = "weights")
            if type(b) != np.ndarray:
                self.b = tf.Variable(tf.random_normal([s]), 
                                     trainable = self.trainable,
                                     name = "biases")
            else:
                self.b = tf.Variable(b, trainable = self.trainable, name = "biases")
            self.y = tf.add(tf.matmul(self.x, self.w), self.b, name = "output")
            self.session.run(tf.global_variables_initializer())
    
    def init_training_vars(self):
        """Initialize training procedure"""
        self._q_targets = tf.placeholder(shape = [None, self.size * self.size], 
                                         dtype = tf.float32)
        self._loss = tf.reduce_sum(tf.square(self._q_targets - self.y))
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate = .1)
        self._update = self._optimizer.minimize(self._loss)

    def train(self, states, targets):
        """Trains a model over a given set of states and targets"""
        # Reshape
        states = np.array([np.reshape(s, -1) for s in states])
        targets = np.array([np.reshape(t, -1) for t in targets])
        # Run training update
        self.session.run(self._update, 
                         feed_dict = {self.x: states, self._q_targets: targets})
    
    def train_rotate(self, states, targets, save = True):
        """Train over the rotated versions of each state and reward"""
        # Reshape targets for rotation
        targets = [np.reshape(t, states[0].shape) for t in targets]
        # Collect rotated versions of each state and target
        newStates = []
        newTargets = []
        for s, t in zip(states, targets):
            ns = s
            nt = t
            for i in range(3):
                rs = self.rotate(ns)
                rt = self.rotate(nt)
                newStates.append(rs)
                newTargets.append(np.reshape(rt, -1))
                ns = rs
                nt = rt
        # Combine lists
        allStates = states + newStates
        allTargets = targets + newTargets
        if save:
            self.states.extend(newStates)
            self.targets.extend(newTargets)        
        # Train
        self.train(allStates, allTargets)
    
    def rotate(self, x):
        """Rotates an array counter-clockwise"""
        n = np.zeros(x.shape)
        for i in range(x.shape[0]):
            n[:, i] = x[i][::-1]
        return n

    def save(self, prefix = "agents/params/"):
        """Save the models parameters in a .npz file"""
        today = datetime.now()
        name = prefix + "QNoHidden_{0}_{1}_{2}_{3}_{4}.npz".format(
            str(today.year)[2:], today.month, today.day, today.hour, today.minute)
        with open(name, "wb") as f:
            np.savez(f, 
                     w = self.w.eval(session = self.session),
                     b = self.b.eval(session = self.session))
    
    def load(self, name, prefix = "agents/params/"):
        """Loads a model from a given .npz file"""
        name = prefix + name
        with open(name, "rb") as f:
            loaded = np.load(f)
            self.init_model(w = loaded["w"], b = loaded["b"])