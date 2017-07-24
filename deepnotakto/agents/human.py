# human.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from agent import Agent

class Human (Agent):
    def __init__(self):
        """
        Initializes a human agent
        """
        # Call parent initializer
        super(Human, self).__init__()
    
    def act(self, env):
        """Choose and action and apply to environment"""
        move = self.get_turn(env)
        if move == False:
            env._end_episode = True
            env._end_training = True
            return None
        zeros = np.zeros(env.shape)
        zeros[move[0], move[1]] = 1
        env.act(zeros)
    
    def get_turn(self, env):
        """Get turn input"""
        while True:
            inp = input("Next Piece: ")
            if inp == "exit":
                return False
            if len(inp) != 3:
                print("Please enter valid position")
                continue
            row, col = [int(i) for i in inp.split()]
            row = int(row) - 1
            col = int(col) - 1
            if row < 0 or col < 0 or row >= env.size or col >= env.size:
                print("Please enter valid position")
                continue
            return [row, col]