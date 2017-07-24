# environment.py
# Abraham Oliver, 2017
# Deep-Nakato Project

import numpy as np
from copy import copy

class Env (object):
    def __init__(self, size):
        """
        Initializes the environment
        Parameters:
            size (Int) - Side length of the board (board size = size * size)
        """
        self.size = size
        self.shape = (size, size)
        self.reset()
    
    def reset(self):
        """Reset board"""
        self.board = np.zeros(self.shape, dtype = np.int32)
        self.turn = 0
        self._end_episode = False
        self._end_training = False
    
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
        new_board = np.add(self.board, action)
        # If illegal move, highly negative reward
        if np.max(new_board) > 1:
            self._end_episode = True
            return -1000
        # Rewards based on winner
        winner = self.is_over(new_board)
        if winner == 0:
            return 0
        else:
            return -100
        
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
        if board != np.ndarray:
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
    
    def forced(self):
        """Is a loss forced on the next turn"""
        b = np.reshape(copy(self.board), -1)
        # Calculate possible moves for opponent
        remaining = []
        for i in range(b.size):
            if b[i] == 0:
                x = copy(b)
                x[i] = 1
                remaining.append(x)
        # If all are losses, a loss is forced
        for r in remaining:
            if self.is_over(np.reshape(r, self.shape)) == 0:
                return False
        return True
    
    def train(self, a1, a2, episodes, display = False, rotate_player_one = False):
        """
        Train two agents over a given number of episodes
        Parameters:
            a1 (Agent) - An agent to train
            a2 (Agent) - An agent to train
            episodes (int) - Number of episodes to train over
            display (bool) - Passed to play function for game output
            rotate_player_one (bool) - Should first turn be rotated
        """
        if not display:
            print("Training .", end = "")
        display_interval = episodes // 20 if episodes >= 20 else 1
        for i in range(episodes):
            # Reset board
            self.reset()
            # Play game
            self.play(a1, a2, display = display)
            # Quit if needed
            if self._end_training:
                print()
                print("Training ended by a human agent")
                return None
            # Display status
            if not display:
                if i % display_interval == 0:
                    print(".", end = "")
            # Rotate players if needed
            if rotate_player_one:
                a1, a2 = a2, a1
        if not display:
            print(" Done")
            
    
    def __str__(self):
        """Conversion to string"""
        print()
        for i in self.board:
            for j in i:
                print("O" if j == 0 else "X", end = " ")
            print()
        return ""
    
    def display(self):
        """Print board"""
        self.__str__()
        print()
        
    def play(self, a1, a2, display = False):
        """
        Plays two agents against eachother
        Parameters:
            a1 (agent.Agent) - Agent for player 1
            a2 (agent.Agent) - Agent for player 2
            display (Bool) - Should debug print board and winner
        Note:
            Currently throws an error if both agents play an illegal move (thus not
            changing the board). This element of the system will be removed once
            agents can be garaunteed not to play illegal moves.
        """
        # Is the game loop finished
        done = False
        # Has the last turn been missed because of an illegal move
        last_missed = False
        # Main game loop
        while not done and not self._end_episode:
            # Copy the board for later comparison pre and post move
            b_copy = copy(self.board) 
            if display:
                print("Turn #{}".format(self.turn))
            # Play the agent corresponding to the current turn
            if self.turn % 2 == 0:
                a1.act(self)
            else:
                a2.act(self)
            # Change turn
            self.turn += 1
            if display:
                self.display()
            
            # Catch double illegal moves
            if np.equal(b_copy, self.board).all():
                if display:
                    print("Player attempted illegal move")
                # If a move was not made, but the previous one was
                if not last_missed:
                    last_missed = True
                # If a move was not made and the last was
                else:
                    raise ValueError("Two missed turns in a row")
            # If a move was made
            else:
                last_missed = False
            
            # End the loop if game is over
            done = False if self.is_over() == 0 else True
        if display:
            print("Player {} Wins!".format(1 if self.turn % 2 == 0 else 2))