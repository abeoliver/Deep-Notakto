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
        self.board = np.zeros(self.shape, dtype = np.int32)
        self.turn = 0
    
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
        # If illegal move, highly negative reward
        if np.max(np.add(self.board, action)) > 1:
            return -1000
        # Rewards based on winner
        winner = self.is_over()
        if winner == 0:
            return 0
        elif winner == 1:
            if self.turn % 2 == 0:
                return 100
            else:
                return -100
        elif winner == 2:
            if self.turn % 2 == 1:
                return 100
            else:
                return -100
        
    def act(self, action):
        """
        Perform an action on the environment
        Parameters:
            action ((N, M) array) - One hot of the desired move
        Note:
            When an illegal move is attempted no move is executed
        """
        # Play the move if the move isn't legal
        if not np.max(np.add(self.board, action)) > 1:
            self.board = np.add(self.board, action)
    
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

    def play_hvh(self):
        """
        Play two humans against eachother
        Note:
            "exit" as input ends the game
        """
        def play_piece(row, col):
            if self.board[row, col] != 0:
                return False
            else:
                self.board[row, col] = 1
                return True
        b = copy(self.board)
        done = False
        self.display()
        while not done:
            print("Player {}".format(1 if self.turn % 2 == 0 else 2))
            inp = input("Next Piece: ")
            if inp == "exit":
                return False
            if len(inp) != 3:
                print("Please enter valid position")
                continue
            row, col = [int(i) for i in inp.split()]
            row = int(row) - 1
            col = int(col) - 1
            if row < 0 or col < 0:
                print("Please enter valid position")
                continue
            played = play_piece(row, col)
            if played:
                self.turn += 1
                self.display()
                done = True if self.is_over() else False
            else:
                print("Please enter ")
        print("GAME OVER! Player {} Wins!".format(1 if self.turn % 2 == 0 else 2))
    
    def play_cvc(self, a1, a2, display = False):
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
        while not done:
            # Copy the board for later comparison pre and post move
            b_copy = copy(self.board) 
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

    def is_over(self):
        """Checks if game is over"""
        b = copy(self.board)
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