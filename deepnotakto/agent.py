# agent.py
# Abraham Oliver, 2017
# Deep-Notakto Project

from numpy import reshape
from numpy.random import normal

class Agent (object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
    
    def act(self, env, **kwargs):
        pass
    
    def flatten(self, state):
        return reshape(state, -1)
    
    def train(self, states, targets):
        pass
    
    def save(self, name = "agent"):
        pass
    
    def show_Q(self, board):
        """Shows confidences for a given board"""
        return None

    def get_Q(self, board):
        """Gets the Q-values (all random)"""
        return normal(size = board.shape)
    
    @staticmethod
    def load(name):
        """Loads a model from a given file name"""
        pass