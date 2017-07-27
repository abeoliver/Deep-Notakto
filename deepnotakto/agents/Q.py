# Q.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from random import choice
from datetime import datetime
from agent import Agent

class Q (Agent):
    def __init__(self, layers, load_file_name = None, gamma = .8, trainable = True,
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
        super(Q, self).__init__()
        self.layers = layers
        self.size = np.int32(np.sqrt(layers[0]))
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
            if type(w) != list:
                self.w = [tf.Variable(tf.random_normal([self.layers[n], self.layers[n + 1]]),
                                      trainable = self.trainable, name = "weights_{}".format(n))
                          for n in range(len(self.layers) - 1)]
            else:
                self.w = [tf.Variable(weight, trainable = self.trainable, name = "weights_{}".format(n))
                          for n, weight in enumerate(w)]
            if type(b) != list:
                self.b = [tf.Variable(tf.random_normal([1, self.layers[n + 1]]),
                                      trainable = self.trainable, name = "biases_{}".format(n))
                          for n in range(len(self.layers) - 1)]
            else:
                self.b = [tf.Variable(bias, trainable = self.trainable, name = "biases_{}".format(n))
                          for n, bias in enumerate(b)]
            # Predicted output
            def feed(inp, n=0):
                """Recursive function for feeding a vector through layers"""
                # End recursion
                if n == len(self.layers) - 2:
                    # Minus 2 because final layer does no math (-1) and the lists start at zero (-1)
                    return tf.matmul(inp, self.w[n], name = "feedmul{0}".format(n)) + self.b[n]
                # Continue recursion
                return feed(tf.matmul(inp, self.w[n], name = "feedmul{0}".format(n)) + self.b[n], n + 1)
            self.y = feed(self.x)
            self.session.run(tf.global_variables_initializer())
    
    def init_training_vars(self):
        """Initialize training procedure"""
        with tf.name_scope("training"):
            self._q_targets = tf.placeholder(shape = [None, self.size * self.size], 
                                             dtype = tf.float32, name = "targets")
            self._loss = tf.reduce_sum(tf.square(self._q_targets - self.y), name = "loss")
            self._optimizer = tf.train.GradientDescentOptimizer(learning_rate = .1, name = "optimizer")
            self._update = self._optimizer.minimize(self._loss, name = "update")

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
        print("Rotating ... ", end = "")
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
        print("Done")
        # Combine lists
        allStates = states + newStates
        allTargets = targets + newTargets
        if save:
            self.states.extend(newStates)
            self.targets.extend(newTargets)        
        # Train
        print("Training ... ", end = "")
        self.train(allStates, allTargets)
        print("Done")
    
    def rotate(self, x):
        """Rotates an array counter-clockwise"""
        n = np.zeros(x.shape)
        for i in range(x.shape[0]):
            n[:, i] = x[i][::-1]
        return n

    def save(self, name = None, prefix = "agents/params/"):
        """
        Save the models parameters in a .npz file
        Parameters:
            name (string) - File name for save file
            prefix (string) - The file path prefix
        """
        today = datetime.now()
        if not name:
            name = prefix + "Q({0})_{1}_{2}_{3}_{4}_{5}.npz".format(
                self.layers, str(today.year)[2:], today.month, today.day, today.hour, today.minute)
        with open(name, "wb") as f:
            np.savez(f, 
                     w = [w.eval(session = self.session) for w in self.w],
                     b = [b.eval(session = self.session) for b in self.b])
    
    def load(self, name, prefix = "agents/params/", trainable = False):
        """Loads a model from a given .npz file"""
        name = prefix + name
        with open(name, "rb") as f:
            loaded = np.load(f)
            self.trainable = trainable
            print(loaded["w"])
            self.init_model(w = loaded["w"], b = loaded["b"])