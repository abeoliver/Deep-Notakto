# agent.py
# Abraham Oliver, 2017
# Deep-Notakto Project
from numpy.random import normal

class Agent (object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.buffer = []
        self.buffer_lengths = []
    
    def act(self, env, **kwargs):
        pass
    
    def train(self, states, targets):
        pass

    def record(self, state, action, reward):
        """Record an action"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def reset_memory(self):
        """Reset the memory of an agent"""
        self.states = []
        self.actions = []
        self.rewards = []

    def add_buffer(self, state, action, reward):
        """Adds a move of a game to a game buffer"""
        self.buffer.append((state, action, reward))

    def save_buffer(self, use_final = False, reward = None):
        """Saves a buffer to the game records"""
        if len(self.buffer) == 0:
            self.buffer_lengths.append(0)
            return None
        final_reward = self.buffer[-1][2]
        for s, a, r in self.buffer:
            self.states.append(s)
            self.actions.append(a)
            if use_final:
                self.rewards.append(final_reward)
            elif reward != None:
                self.rewards.append(reward)
            else:
                self.rewards.append(r)
        self.buffer_lengths.append(len(self.buffer))
        self.buffer = []

    def reset_buffer(self):
        self.buffer = []

    def get_last_buffer(self):
        i = self.buffer_lengths[-1]
        return (self.states[-i:], self.actions[-i:], self.rewards[-i:])
    
    def save(self, name = "agent"):
        pass
    
    def show_Q(self, board):
        """Shows confidences for a given board"""
        return None

    def get_Q(self, board):
        """Gets the Q-values (all random)"""
        return normal(size = board.shape)