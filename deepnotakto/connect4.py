#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: notakto_treesearch.py                                        #
#  Abraham Oliver, 2018                                               #
#######################################################################
import sys
from copy import copy
from random import choice

import numpy as np

from deepnotakto.environment import Env as BaseEnv
from deepnotakto.agents import Agent as BaseAgent

class Env (BaseEnv):
    def __init__(self, width = 7, height = 6, starting = None, rewards = None):
        """
        Initializes the environment
        Args:
            width: (int) Number of columns
            height: (int) Height of columns
            starting: (array) Starting board configuration if not blank
            rewards: (dict) Custom reward values
        """
        self.width = width
        self.height = height
        self.shape = (height, width)
        if starting is None:
            self.starting = np.zeros(self.shape, dtype = int)
        else:
            self.starting = starting
        self.starting_turn = self.without_players(self.starting).sum()
        self.reset()
        if rewards is None:
            self.rewards = {
                "illegal": -10,
                "win": 2,
                "loss": -2
            }
        else:
            self.rewards = rewards

    def without_players(self, array):
        return _without_players(array)

    def current_player(self, state = None):
        if state is None:
            state = self.state
        return _current_player(state)

    def legal(self, action, state = None):
        if state is None:
            state = self.state
        return _legal(action, state)

    def reward(self, action, state = None):
        """
        Returns the immediate reward for a given action

        Args:
            action: (action) Action to play on game state
        Returns:
            (int) Reward for given action
        """
        if state is None:
            state = self.state
        # If illegal move, highly negative reward
        if self.legal(action, state):
            return self.rewards["illegal"]
        # Play the move on a copy of the board
        new_state = self.play_move_on_state(state, action)

        # Rewards based on winner
        winner = self.winner(new_state)
        if winner == 0:
            return 0
        elif winner == self.current_player(state):
            return self.rewards["win"]
        else:
            return self.rewards["loss"]

    def play_move_on_state(self, state, action):
        self.state = _play_move_on_state(
            state, action, self.current_player(state)
        )

    def winner(self, state = None):
        if state is None:
            state = self.state
        return _winner(state)

    def action_space(self, state = None):
        """
        Returns a list of all possible moves (reguardless of win / loss)

        Args:
            board: (array) Current board state
        Returns:
            (array[]) All legal moves for the given board
        """
        if state is None:
            state = self.state
        return _action_space(state)

    def __str__(self):
        return _print_board(self.state)


class RandomAgent (BaseAgent):
    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = "Random"

    def get_action(self, state):
        possible = _action_space(state)
        player = _current_player(state)
        opponent = 2 if player == 1 else 1
        not_loser = []
        # Choose a winner and identify non-losers
        for m in possible:
            # Make move temporarily
            new_state = _play_move_on_state(copy(state), m, player)
            winner = _winner(new_state)
            # If it is a loser, do not consider
            if winner == opponent:
                continue
            elif winner == player:
                return m
            # If neither, remember that it is a not a losing move
            not_loser.append(m)
        if len(not_loser) >= 1:
            # If there are non-losing move, choose one randomly
            return choice(not_loser)
        else:
            # If all moves are losses, choose any
            return choice(possible)


class Human (BaseAgent):
    def __init__(self):
        """ Initializes a human agent """
        # Call parent initializer
        super(Human, self).__init__()
        self.name = "Human"

    def get_action(self, state):
        """ Get the action from the user """
        # Continue prompting until a valid move is made
        while True:
            # Prompt for user choice
            inp = input("Column to play: ")
            # Exit program if human desires
            if inp == "exit":
                sys.exit()
            try:
                move = int(inp) - 1
            except ValueError:
                print("Not a valid column")
                continue
            if not _legal(state, move):
                print("Not a valid column")
                continue
            return move


def play(a1, a2, games = 1, env_args = {}, clear_func = None):
    played_games = 0
    players = [a1, a2]
    while played_games < games or games == -1:
        e = Env(**env_args)
        while True:
            if clear_func is not None:
                clear_func()
            _print_board(e.state)
            move = players[e.current_player() - 1].get_action(e.state)
            e.play_move_on_state(e.state, move)
            winner = e.winner()
            if winner == 1:
                print("============= Player 1 wins! =============\n\n")
                break
            elif winner == 2:
                print("============= Player 2 wins! =============\n\n")
                break
        played_games += 1


# Utility functions
def _action_space(state):
    available = []
    for i in range(state.shape[1]):
        if state[:, i].sum() < state.shape[0]:
            available.append(i)
    return available

def _without_players(array):
    return np.not_equal(array, 0).astype(int)

def _current_player(state):
    pieces = _without_players(state).sum()
    # If there are an even number of pieces, it is player 1's turn
    if pieces % 2 == 0:
        return 1
    # Odd number of pieces, 2's turn
    return 2

def _legal(state, action):
    if action >= state.shape[1]:
        return False
    # Get the desired column in terms of empty / full spaces
    simple_col = _without_players(state[:, action])
    # If completely full, raise exception
    if np.equal(simple_col, 1).all():
        return False
    return True

def _play_move_on_state(state, action, piece):
    # NOTE action should already be checked for legal move
    # Copy the state to avoid pythonic accidental overwrites
    state = state.copy()
    # Get the desired column in terms of empty / full spaces
    simple_col = _without_players(state[:, action])
    # If completely empty, fill last slot
    if np.equal(simple_col, 0).all():
        top_of_column = state.shape[0] - 1
    # Otherwise, stack in column
    else:
        top_of_column = simple_col.argmax() - 1
    # Actually play the piece
    state[top_of_column, action] = piece
    # Return the new state
    return state

def _print_board(state, col_nums = True):
    """ Conversion to string """
    out = ""
    if col_nums:
        for i in range(state.shape[1]):
            out += str(i + 1)
            out += " "
        out += "\n"
    for i in state:
        for j in i:
            if j == 1:
                out += "O"
            elif j == 2:
                out += "X"
            else:
                out += "*"
            out += " "
        out += "\n"
    print(out)

# Functions for finding winners of Connect4
filters = [
    np.identity(4),
    np.fliplr(np.identity(4)),
    np.ones((1,4)).T,
    np.ones((1,4))
]

def convolve(x, k):
    out_rows = x.shape[0] - k.shape[0] + 1
    out_cols = x.shape[1] - k.shape[1] + 1
    out = np.zeros((out_rows, out_cols))
    for c in range(out_cols):
        for r in range(out_rows):
            out[r,c] = np.multiply(
                x[r:r + k.shape[0], c: c + k.shape[1]], k
            ).sum()
    return out

def only_player(x, p):
    return np.equal(x, p).astype(int)

def does_player_win(x, p):
    p_board = only_player(x, p)
    for f in filters:
        if convolve(p_board, f).max() >= 4:
            return True
    return False

def _winner(state):
    if does_player_win(state, 1):
        return 1
    if does_player_win(state, 2):
        return 2
    return 0

def main():
    import os
    r = RandomAgent()
    h = Human()
    play(h,r, -1, clear_func = lambda: os.system('clear'))

if __name__ == "__main__":
    main()
