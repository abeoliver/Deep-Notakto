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
        # Choose a winner and identify non-losers
        for m in possible:
            new_state = add(m, state)
            # Make move if force (winner)
            if env.forced(new_state):
                move = m
                break
            winner = env.is_over(new_state)
            if winner == 0:
                not_loser.append(m)
        if not equal(move, 0).all():
            pass
        elif len(not_loser) >= 1:
            move = choice(not_loser)
        else:
            move = choice(possible)
        # Make move
        _, reward = env.act(move)
        self.record(state, move, reward)
        return [state, move, reward]
