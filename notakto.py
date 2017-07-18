# natakto.py
# Abraham Oliver, 2017
# Deep-Nakato Project

import numpy as np
from copy import copy

class Board (object):
    def __init__(self, size):
        self.size = size
        self.board = np.zeros([size, size], dtype = np.int32)
    
    def __str__(self):
        print()
        for i in self.board:
            for j in i:
                print("O" if j == 0 else "X", end = " ")
            print()
        return ""
    
    def display(self):
        self.__str__()
        print()

    def play_piece(self, row, col):
        if self.board[row, col] != 0:
            return False
        else:
            self.board[row, col] = 1
            return True

    def play_hvh(self):
        b = copy(self.board)
        done = False
        self.display()
        turn = 0
        while not done:
            print("Player {}".format(1 if turn % 2 == 0 else 2))
            inp = input("Next Piece: ")
            if len(inp) != 3:
                print("Please enter valid position")
                continue
            row, col = [int(i) for i in inp.split()]
            if row == "x" or col == "x":
                return False
            row = int(row) - 1
            col = int(col) - 1
            if row < 0 or col < 0:
                print("Please enter valid position")
                continue
            played = self.play_piece(row, col)
            if played:
                turn += 1
                self.display()
                done = self.is_over()
            else:
                print("Please enter ")
        print("GAME OVER! Player {} Wins!".format(1 if turn % 2 == 0 else 2))
    
    def play_hvc(self):
        

    def is_over(self):
        b = copy(self.board)
        # Rows
        for row in b:
            if np.sum(row) == b.shape[0]:
                return True
        # Columns (row in transpose of b)
        for col in b.T:
            if np.sum(col) == b.shape[0]:
                return True
        # Diagonals
        # Top left to bottom right
        tlbr = copy(b) * np.identity(self.size) * 1000
        if np.sum(tlbr) >= 1000 * self.size:
            return True
        # Bottom left to top right
        bltr = copy(b) * np.flip(np.identity(self.size), 1) * 1000
        if np.sum(bltr) >= 1000 * self.size:
            return True
        # Otherwise game is not over
        return False