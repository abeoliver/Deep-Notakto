# rl_3_no_hidden.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from agent import Agent

class RLNoHidden (Agent):
    def __init__(self, size, load_file_name = None):
        """
        Initializes an agent
        Parameters:
            size (int) - Board side length
            load_file_name (string) - Path to load saved model from
        Note:
            Initializes randomly if no model is given
        """
        # Call parent initializer
        super(RLNoHidden, self).__init__()
        self.size = size
        # Create a tensorflow session for all processes to run in
        self.session = tf.Session()
        # Load model if a file name is given
        if load_file_name != None:
            self.load_model(load_file_name)
        # Otherwise randomly initialize
        else:
            self.init_model()
        
    def act(self, env):
        """
        Choose action, apply action to environment, and recieve reward
        Parameters:
            env (environment.Env) - Environment of the agent
        """
        # Current environment state
        state = env.observe()
        self.states.append(state)
        # Get action probabilties
        probs = self.get_probs(state)
        # Make a blank action
        action = np.zeros(probs.shape, dtype = np.int32)
        # Make a stochastic decision
        max_index = np.argmax(probs)
        action[max_index] = 1
        # Add action to action history
        self.actions.append(move)
        # Reshape action for applying
        action = np.reshape(action, state.shape)
        # Apply action, add reward to reward history
        self.rewards.append(env.act(action))
    
    def get_probs(self, state):
        """
        Get action probabilities
        Parameters:
            state ((N, N) array) - Current environment state
        """
        # Pass the state to the model and get array of probabilities
        return self.y.eval(session = self.session, 
                           feed_dict = {self.x: [self.flatten(state)]})[0]

    def init_model(self):
        """Randomly intitialize model"""
        self.x = tf.placeholder(tf.float32, [None, self.size])
        self.w = tf.constant(tf.random_normal([self.size, self.size]))
        self.bias = tf.constant(tf.random_normal([self.size]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.w) + self.bias)
        