# agent.py
# Abraham Oliver, 2017
# Deep-Notakto Project

from numpy import reshape

class Agent (object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
    
    def act(self, env):
        pass
    
    def flatten(self, state):
        return reshape(state, -1)
    
    def train(self, states, targets):
        pass