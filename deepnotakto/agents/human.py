# human.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
from agent import Agent
import sys

class Human (Agent):
    def __init__(self):
        """Initializes a human agent"""
        # Call parent initializer
        super(Human, self).__init__()
        self.name = "Human"
    
    def get_action(self, state):
        """Get the action from the user"""
        # Continue prompting until a valid move is made
        while True:
            # Prompt for user choice
            inp = input("Next Piece: ")
            # Exit program if human desires
            if inp == "exit":
                sys.exit()
            # Split move into [row, column]
            row, col = inp.split()
            if type(row) != int or row > state.shape[0] or row < 1:
                print("Please enter valid position")
                continue
            elif type(col) != int or col > state.shape[0] or col < 1:
                print("Please enter valid position")
                continue
            row = int(row) - 1
            col = int(col) - 1
            action = np.zeros(state.shape, dtype = np.int32)
            action[row, column] = 1
            return action