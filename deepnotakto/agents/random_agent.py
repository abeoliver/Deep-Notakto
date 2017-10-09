# random_agent.py
# Abraham Oliver, 2017
# Deep-Notakto Project

from numpy import zeros, int32, add, equal, int32
from random import randint, choice
from agents.agent import Agent

class RandomAgent (Agent):
    def __init__(self, env):
        super(RandomAgent, self).__init__()
        self.name = "RandomPlus"
        self.env = env

    def get_action(self, state):
        possible = self.env.action_space(state)
        player = 2 if self.env.turn % 2 == 0 else 1
        not_loser = []
        move = zeros(state.shape, int32)
        winning_move = False
        # Choose a winner and identify non-losers
        for m in possible:
            # Make move temporarily
            new_state = add(m, state)
            # Discard if it is a loss
            winner = self.env.is_over(new_state)
            if winner != 0:
                continue
            # If forced loss on opponent, choose move
            if self.env.forced(new_state):
                move = m
                winning_move = True
                break
            # If neither, remember that it is a not a losing move
            not_loser.append(m)
        if winning_move:
            # Don't choose new move if move already chosen
            pass
        elif len(not_loser) >= 1:
            # If there are non-losing move, choose one randomly
            move = choice(not_loser)
        else:
            # If all moves are losses, choose any
            move = choice(possible)
        return move
