# environment.py
# Abraham Oliver, 2017
# Deep-Nakato Project

import numpy as np
from copy import copy
import matplotlib.pyplot as plt

"""
Define "Observation" (dict)
Keys:
    "observation" (nd-array) - Board state
    "reward" (float) - Reward for a given action
    "done" (bool) - Is at terminal state?
    "info" (dict) - Other environment information
"""

class Env (object):
    def __init__(self, size, rewards = None):
        """
        Initializes the environment
        Parameters:
            size (Int) - Side length of the board (board size = size * size)
        """
        # Board variables
        self.size = size
        self.shape = (size, size)
        self.reset()
        if rewards == None:
            self.rewards = {
                "illegal": -10,
                "forced": 2,
                "loss": -2
            }
        else:
            self.rewards = rewards
    
    def reset(self):
        """Reset board"""
        self.board = np.zeros(self.shape, dtype = np.int32)
        self.turn = 0
        self._illegal = False
    
    def observe(self):
        """The observe step of the reinforcement learning pipeline"""
        return copy(self.board)
    
    def reward(self, action):
        """
        Returns the immediate reward for a given action
        Parameters:
            action ((N, N) array) - One hot of the given move
        Returns:
            Int - Reward for given action
        """
        new_board = np.add(self.board, action)
        # If illegal move, highly negative reward
        if np.max(new_board) > 1:
            self._illegal = True
            return self.rewards["illegal"]
        # Rewards based on winner
        winner = self.winner(new_board)
        if winner == 0:
            # Positive reward for forcing a loss
            if self.forced(new_board):
                # High reward for a forced win
                return self.rewards["forced"]
            # No reward
            return 0
        else:
            # Negative reward for a loss
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
        """
        # Calculate move reward
        reward = self.reward(action)
        # Calculate move effect
        moved = np.add(self.board, action)
        # Play the move if the move isn't legal
        if np.max(moved) > 1:
            illegal = True
        else:
            self.board = moved
            illegal = False
            # Update turn counter
            self.turn += 1
        return {
            "observation": self.board,
            "reward": reward,
            "done": self.winner(),
            "action": action,
            "info": {"illegal": illegal}
        }
    
    def winner(self, board = None):
        """Checks if game is over"""
        if type(board) == type(None):
            board = self.board
        # Rows
        for row in board:
            if np.sum(row) == board.shape[0]:
                return 1 if self.turn % 2 == 0 else 2
        # Columns (row in transpose of b)
        for col in board.T:
            if np.sum(col) == board.shape[0]:
                return 1 if self.turn % 2 == 0 else 2
        # Diagonals
        # Top left to bottom right
        tlbr = np.sum(board * np.identity(self.size))
        if tlbr >= self.size:
            return 1 if self.turn % 2 == 0 else 2
        # Bottom left to top right
        bltr = np.sum(board * np.flip(np.identity(self.size), 1))
        if bltr >= self.size:
            return 1 if self.turn % 2 == 0 else 2
        # Otherwise game is not over
        return 0
    
    def forced(self, board = None):
        """Is a loss forced on the next turn"""
        if type(board) != np.ndarray:
            b = copy(self.board)
        else:
            b = copy(board)
        # If (n-1)^2 + 1 pieces are played, then garaunteed force
        if np.sum(b) > (b.shape[0] - 1) ** 2:
            return True
        # Calculate possible moves for opponent
        remaining = self.action_space(b)
        # If all are losses, a loss is forced
        for r in remaining:
            if self.winner(np.add(b, r)) == 0:
                return False
        return True
    
    def action_space(self, board = None):
        """
        Returns a list of all possible moves (reguardless of win / loss)
        Parameters:
            board ((N, N) array) - Current board state (default self.board)
        Returns:
            List of (N, N) arrays - AllActivated legal moves for the given board
        """
        # Get board
        if type(board) != np.ndarray:
            b = copy(self.board)
        else:
            b = copy(board)
        remaining = []
        # Loop over both axes
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                # If there is an empty space, add the move to remaining moves
                if b[i, j] == 0:
                    z = np.zeros(b.shape, dtype = np.int32)
                    z[i, j] = 1
                    remaining.append(z)
        return remaining
    
    def __str__(self):
        """Conversion to string"""
        print()
        for i in self.board:
            for j in i:
                print("O" if j == 0 else "X", end = " ")
            print()
        return ""
    
    def display(self):
        """Prints board or shows it as an image"""
        self.__str__()
        print()