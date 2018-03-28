#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: agents/perfect.py                                            #
#  Abraham Oliver, 2018                                               #
#######################################################################

from random import choice

import numpy as np
import util
from agent import Agent


class Perfect (Agent):
    def __init__(self, move_file, size):
        """Initializes a perfect agent (only works for the player the file intends)"""
        # Call parent initializer
        super(Perfect, self).__init__()
        self.name = "Perfect"
        self.moves = util.get_move_dict(move_file, size)

    def get_action(self, state):
        # Convert the board into a move-dict key
        b = util.array_to_bin(state)
        # Get the perfect moves
        possible = self.moves[b]
        # Check how many perfect moves there are
        s = np.sum(possible)
        if s == 1:
            # If only one, the possible move is the same as the perfect move array
            return possible
        else:
            # If there are more than one perfect moves, choose one
            # Get a random argument for a perfect move
            arg = choice(possible.argsort(None)[-s:].tolist())
            action = np.zeros([state.size], dtype = np.int32)
            action[arg] = 1
            return np.reshape(action, state.shape)