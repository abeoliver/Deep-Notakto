# rl_3_no_hidden.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from agent import Agent

class RL3NoHidden (Agent):
    def __init__(self):
        self.session = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, 9])
        self.w = tf.Variable(tf.random_normal([9, 9]))
        self.bias = tf.Variable(tf.ones([9]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.w) + self.bias)
        # Initialize variables
        self.session.run(tf.global_variables_initializer())
        
    def play(self, board):
        probs = self.y.eval(session = self.session,
                            feed_dict = {self.x: [board.flatten()]})
        # LOOKUP NON MAXIMUM SUPRESSION
        probs[probs == np.max(probs)] = 1
        probs[probs != 1] = 0
        return np.reshape(probs, board.shape)