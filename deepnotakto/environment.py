# environment.py
# Abraham Oliver, 2017
# Deep-Nakato Project

import numpy as np
from copy import copy
import matplotlib.pyplot as plt

class Env (object):
    def __init__(self, size, win_reward = 20):
        """
        Initializes the environment
        Parameters:
            size (Int) - Side length of the board (board size = size * size)
        """
        # Board variables
        self.size = size
        self.shape = (size, size)
        self.reset()
        # Reward variables
        self.win_reward = win_reward
    
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
            return -10
        # Rewards based on winner
        winner = self.is_over(new_board)
        if winner == 0:
            # Positive reward for forcing a loss
            # Not possible with three or fewer moves so don't check
            if self.turn >= 2:
                if self.forced(new_board):
                    # High reward for a forced win
                    return 8
                else:
                    # Small reward for lasting long
                    return 1
            # No reward
            return 0
        else:
            # Negative reward for a loss
            return -5
        
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
    
    def train(self, a1, a2, episodes, display = False, rotate_player_one = False,
             learn_rate = .01, continuous_update = False, image = False,
             confidences = False):
        """
        Train two agents over a given number of episodes
        Parameters:
            a1 (Agent) - An agent to train
            a2 (Agent) - An agent to train
            episodes (int) - Number of episodes to train over
            display (bool) - Passed to play function for game output
            rotate_player_one (bool) - Should first turn be rotated
            learn_rate (float) - Learning rate for training
            continuous_update (bool) - Update model continuously or not
            image (bool) - Display board as image or not
            confidences (bool) - Show confidences of each AI move
        """
        if not display:
            print("Training ", end = "")
        display_interval = episodes // 10 if episodes >= 10 else 1
        for i in range(episodes):
            # Reset board
            self.reset()
            # Play game
            winner = self.play(a1, a2, display = display, training = continuous_update,
                              learn_rate = learn_rate, image = image, confidences = confidences)
            # Quit if needed
            if self._end_training:
                print()
                print("Ended by a human agent")
                return None
            # Give small reward for winning to winning player if reward is zero
            if winner == 1:
                if a1.rewards[-1] == 0:
                    a1.rewards[-1] = self.win_reward
            if winner == 2:
                if a2.rewards[-1] == 0:
                    a2.rewards[-1] = self.win_reward   
            # Display status
            if not display:
                if i % display_interval == 0:
                    print("*", end = "")
            # Rotate players if needed
            if rotate_player_one:
                a1, a2 = a2, a1
        if not display:
            print(" Done")
    
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
    
    def display(self, image = False):
        """Prints board or shows it as an image"""
        if not image:
            self.__str__()
            print()
        else:
            plt.imshow(self.observe(), cmap = "gray")
            plt.show()
        
    def play(self, a1, a2, display = False, training = True, learn_rate = .01,
             image = False, confidences = False):
        """
        Plays two agents against eachother
        Parameters:
            a1 (agent.Agent) - Agent for player 1
            a2 (agent.Agent) - Agent for player 2
            display (bool) - Should debug print board and winner
            training (bool) - Is training or not
            learn_rate (float) - Learn rate for training, unused if not training
            image (bool) - Display board as image or not
            confidences (bool) - Show confidences of each AI move
        Note:
            Currently throws an error if both agents play an illegal move (thus not
            changing the board). This element of the system will be removed once
            agents can be garaunteed not to play illegal moves.
        """
        # Is the game loop finished
        done = False
        # Main game loop
        if display:
                self.display(image)
        while not done and not self._end_episode:
            # Copy the board for later comparison pre and post move
            b_copy = copy(self.board) 
            if display:
                print("Turn #{}".format(self.turn))
            # Play the agent corresponding to the current turn
            if self.turn % 2 == 0:
                if confidences:
                    print("Player 1 Confidences")
                    print(a1.show_Q(self.observe()))
                a1.act(self, training = training, learn_rate = learn_rate)
            else:
                if confidences:
                    print("Player 2 Confidences")
                    print(a2.show_Q(self.observe()))
                a2.act(self, training = training, learn_rate = learn_rate)
            # Change turn
            self.turn += 1
            if display:
                self.display(image)
            
            # Catch double illegal moves
            if np.equal(b_copy, self.board).all() and self.turn != 0:
                if display:
                    print("Agent attempted an illegal move")
                done = True
            # End the loop if game is over
            done = False if self.is_over() == 0 else True
        if display:
            self.display(image = image)
            print("Player {} Wins!".format(1 if self.turn % 2 == 0 else 2))
        # Return winner
        return (1 if self.turn % 2 == 0 else 2) if self.is_over() != 0 else 0