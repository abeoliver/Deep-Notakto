# random_agent.py
# Abraham Oliver, 2017
# Deep-Notakto Project

from numpy import zeros, int32
from random import randint
from agent import Agent

class RandomAgent (Agent):
    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = "Random"
    def act(self, env, **kwargs):
        state =  env.observe()
        m = zeros(env.board.shape, dtype = int32)
        r, c = [0, 0]
        while state[r, c] != 0:
            r = randint(0, env.board.shape[0] - 1)
            c = randint(0, env.board.shape[0] - 1)
        m[r, c] = 1
        _, reward = env.act(m)
        self.record(state, m, reward)
        return [state, m, reward]