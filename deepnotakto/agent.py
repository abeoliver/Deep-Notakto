# agent.py
# Abraham Oliver, 2017
# Deep-Notakto Project

from numpy import reshape, int32

class Agent (object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
    
    def act(self, env):
        pass
    
    def flatten(self, state):
        return int32(reshape(state, -1))