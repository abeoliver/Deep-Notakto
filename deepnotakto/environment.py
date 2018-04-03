#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: environment.py                                               #
#  Abraham Oliver, 2018                                               #
#######################################################################

from copy import copy

import numpy as np

"""
Define "Observation" (dict)
Keys:
    "observation" (nd-array) - Board state
    "reward" (float) - Reward for a given action
    "done" (bool) - Is at terminal state?
    "info" (dict) - Other environment information
"""


class Env (object):
    def __init__(self, starting = None, rewards = None):
        """
        Initializes the environment
        Args:
            starting: (state) Starting board configuration
            rewards: (dict) Custom reward values
        """
        if starting is None:
            self.starting = None
        else:
            self.starting = starting
        self.starting_turn = 0
        self.reset()
        if rewards is None:
            self.rewards = {
                "illegal": -10,
                "win": 2,
                "loss": -2
            }
        else:
            self.rewards = rewards
    
    def reset(self):
        """Reset board"""
        self.state = copy(self.starting)
        self.turn = self.starting_turn
    
    def observe(self):
        """ The observe step of the reinforcement learning pipeline """
        return copy(self.state)
    
    def reward(self, action):
        """
        Returns the immediate reward for a given action
        Args:
            action: (action) Action to play on game state
        Returns:
            (int) Reward for given action
        """
        # Play the move on a copy of the board
        new_state = self.play_move_on_state(self.state, action)
        # If illegal move, highly negative reward
        if self.is_illegal(new_state):
            return self.rewards["illegal"]

        # Rewards based on winner
        winner = self.winner(new_state)
        if winner == 0:
            return 0
        if winner == self.player:
            return self.rewards["win"]
        else:
            return self.rewards["loss"]
        
    def act(self, action):
        """
        Perform an action on the environment
        Parameters:
            action ((N, M) array) - One hot of the desired move
        Returns:
            Observation Object
        Note:
            When an illegal move is attempted no move is executed
            and the turn counter is not incremented
        """
        # Calculate move reward
        reward = self.reward(action)
        # Calculate move effect
        moved = self.play_move_on_state(self.state, action)
        # Play the move if the move isn't legal
        if not self.illegal():
            illegal = True
        else:
            self.state = moved
            illegal = False
            # Update turn counter
            self.turn += 1
        return {
            "observation": self.observe(),
            "reward": reward,
            "winner": self.winner(),
            "action": action,
            "info": {"illegal": illegal}
        }

    def play_move_on_state(self, state, action):
        """ Play an action onto a given state """
        return state
    
    def winner(self, state = None):
        """Checks if game is over"""
        return 0
    
    def display(self):
        """ Prints board to output """
        self.__str__()
        print()

    @property
    def player(self):
        """ Get the number of the player currently playing """
        return self.turn % 2 + 1
