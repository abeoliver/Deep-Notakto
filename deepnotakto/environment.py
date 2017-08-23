# environment.py
# Abraham Oliver, 2017
# Deep-Nakato Project

import numpy as np
from copy import copy
import matplotlib.pyplot as plt

class Env (object):
    def __init__(self, size):
        """
        Initializes the environment
        Parameters:
            size (Int) - Side length of the board (board size = size * size)
        """
        # Board variables
        self.size = size
        self.shape = (size, size)
        self.reset()
    
    def reset(self):
        """Reset board"""
        self.board = np.zeros(self.shape, dtype = np.int32)
        self.turn = 0
        self._end = False
        self._illegal = False
    
    def observe(self):
        """The observe step of the reinforcement learning pipeline"""
        return copy(self.board)
    
    def reward(self, action):
        """
        Returns the immediate reward for a given action
        Parameters:
            action ((N, M) array) - One hot of the given move
        Returns:
            Int - Reward for given action
        """
        rewards = {
            "illegal": -100,
            "forced": 8,
            "loss": -5
        }
        new_board = np.add(self.board, action)
        # If illegal move, highly negative reward
        if np.max(new_board) > 1:
            self._illegal = True
            return rewards["illegal"]
        # Rewards based on winner
        winner = self.is_over(new_board)
        if winner == 0:
            # Positive reward for forcing a loss
            # Not possible with three or fewer moves so don't check
            if self.turn >= 2:
                if self.forced(new_board):
                    # High reward for a forced win
                    return rewards["forced"]
            # No reward
            return 0
        else:
            # Negative reward for a loss
            return rewards["loss"]
        
    def act(self, action):
        """
        Perform an action on the environment
        Parameters:
            action ((N, M) array) - One hot of the desired move
        Returns:
            [array, float] - Board state, reward
        Note:
            When an illegal move is attempted no move is executed
        """
        # Calculate move reward
        reward = self.reward(action)
        # Calculate move effect
        move = np.add(self.board, action)
        # Play the move if the move isn't legal
        if not np.max(move) > 1:
            self.board = move
        return (self.board, reward)
    
    def is_over(self, board = None):
        """Checks if game is over"""
        if type(board) != np.ndarray:
            b = copy(self.board)
        else:
            b = copy(board)
        # Rows
        for row in b:
            if np.sum(row) == b.shape[0]:
                return 1 if self.turn % 2 == 0 else 2
        # Columns (row in transpose of b)
        for col in b.T:
            if np.sum(col) == b.shape[0]:
                return 1 if self.turn % 2 == 0 else 2
        # Diagonals
        # Top left to bottom right
        tlbr = copy(b) * np.identity(self.size) * 1000
        if np.sum(tlbr) >= 1000 * self.size:
            return 1 if self.turn % 2 == 0 else 2
        # Bottom left to top right
        bltr = copy(b) * np.flip(np.identity(self.size), 1) * 1000
        if np.sum(bltr) >= 1000 * self.size:
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
        remaining = self.possible_moves(b)
        # If all are losses, a loss is forced
        for r in remaining:
            if self.is_over(np.add(b, r)) == 0:
                return False
        return True
    
    def possible_moves(self, board = None):
        """Returns a list of all possible moves (reguardless of win / loss)"""
        # Get board
        if type(board) != np.ndarray:
            b = copy(self.board)
        else:
            b = copy(board)
        remaining = []
        # If in a 2D shape
        if len(b.shape) == 2:
            for i in range(b.shape[0]):
                for j in range(b.shape[1]):
                    if b[i, j] == 0:
                        z = np.zeros(b.shape, dtype = np.int32)
                        z[i, j] = 1
                        remaining.append(z)
        # If in a 1D shape
        else:
            for i in range(b.shape[0]):
                if b[i] == 0:
                    z = np.zeros(b.shape)
                    z[i] = 1
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
        
    def play(self, a1, a2, games = 1, trainer_a1 = None, trainer_a2 = None,
             display = False, server_display = False, final_reward = False,
             silence = False):
        """
        Plays two agents against eachother
        Parameters:
            a1 (agent.Agent) - Agent for player 1
            a2 (agent.Agent) - Agent for player 2
            games (int) - Number of games to play
            display (bool) - Should debug print board and winner
            trainer_a1 ((N, N) array, (N, N) array, float -> None) -
                Agent #1 online training func
            trainer_a2 ((N, N) array, (N, N) array, float -> None) -
                Agent #2 online training func
            final_reward (bool) - Fill in rewards on buffer with final reward?
        """
        # Initialize
        episode_train_a1, episode_train_a2 = False, False
        if trainer_a1 != None:
            episode_train_a1 = trainer_a1._type == "episode"
        if trainer_a2 != None:
            episode_train_a2 = trainer_a2._type == "episode"
        played_games = 0
        a1.reset_buffer()
        a2.reset_buffer()
        self._end = False
        if not display and not silence:
            if server_display:
                print("Playing Training Games...")
            else:
                print("Playing ", end = "")
            display_interval = games // 10 if games > 10 else 1
        # ---------- GAME SET LOOP ----------
        while played_games < games and not self._end:
            self.reset()
            played_games += 1
            # Is the game loop finished
            done = False
            illegal = False
            # ---------- Main game loop ----------
            if not display and not server_display and not silence:
                if played_games % display_interval == 0:
                    print("*", end = "")
            if display and not silence:
                    self.display()
            # Play while user has not quit and players play legally
            while not done and not self._end and not self._illegal:
                # Copy the board for later comparison pre and post move
                b_copy = copy(self.board)
                if display and not silence:
                    print("Turn #{}".format(self.turn))
                # Play the agent corresponding to the current turn
                if self.turn % 2 == 0:
                    # Run the turn
                    state, action, reward = a1.act(self)
                    # End if user ends
                    if self._end: return None
                    # Train if training is enabled
                    if trainer_a1 != None and not episode_train_a1:
                        trainer_a1(state, action, reward)
                else:
                    # Run the turn
                    state, action, reward = a2.act(self)
                    # End if user ends
                    if self._end: return None
                    # Train if training is enabled
                    if trainer_a2 != None and not episode_train_a2:
                        trainer_a2(state, action, reward)
                # Change turn
                self.turn += 1
                if display:
                    self.display()

                # Catch double illegal moves and end the game if they exist
                if np.equal(b_copy, self.board).all() and self.turn != 0:
                    if display and not silence:
                        print("Agent attempted an illegal move")
                    done = True
                    self._illegal = True
                else:
                    # End the loop if game is over
                    done = False if self.is_over() == 0 else True
            # ---------- End Main game loop ----------
            # Save game buffers
            a1.save_buffer(final_reward)
            a2.save_buffer(final_reward)
            # Train by episode
            if trainer_a1 != None and episode_train_a1:
                ep_s, ep_a, ep_r, = a1.get_last_buffer()
                trainer_a1(ep_s, ep_a, ep_r)
            if trainer_a2 != None and episode_train_a2:
                ep_s, ep_a, ep_r, = a2.get_last_buffer()
                trainer_a2(ep_s, ep_a, ep_r)

            # Show the final board if desired
            if display and not self._illegal and not silence:
                self.display()
                print("Player {} Wins!".format(1 if self.turn % 2 == 0 else 2))
        # Server display
        if not display and not server_display and not silence:
            print(" Done")
        # ---------- END GAME SET LOOP ----------