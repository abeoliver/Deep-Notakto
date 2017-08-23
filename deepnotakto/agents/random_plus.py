# random_plus.py
# Abraham Oliver, 2017
# Deep-Notakto Project

from numpy import zeros, int32, add, equal
from random import randint
from agent import Agent
from random import choice

class RandomAgentPlus (Agent):
    def __init__(self):
        super(RandomAgentPlus, self).__init__()
        self.name = "RandomPlus"

    def act(self, env, **kwargs):
        state =  env.observe()
        possible = env.possible_moves()
        player = 2 if env.turn % 2 == 0 else 1
        not_loser = []
        move = zeros(env.shape)
        winning_move = False
        # Choose a winner and identify non-losers
        for m in possible:
            # Make move temporarily
            new_state = add(m, state)
            # Discard if it is a loss
            winner = env.is_over(new_state)
            if winner != 0:
                continue
            # If forced loss on opponent, choose move
            if env.forced(new_state):
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
        # Make move
        _, reward = env.act(move)
        self.add_buffer(state, move, reward)
        return [state, move, reward]
