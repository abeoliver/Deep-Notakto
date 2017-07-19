# random_agent.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
from random import randint
from agent import Agent

class RandomAgent (Agent):
    def play(self, board):
        m = np.zeros(board.shape)
        r, c = [0, 0]
        while board[r, c] != 0:
            r = randint(0, board.shape[0] - 1)
            c = randint(0, board.shape[0] - 1)
        m[r, c] = 1
        return m