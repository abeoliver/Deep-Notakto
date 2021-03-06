#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: notakto_treesearch.py                                        #
#  Abraham Oliver, 2018                                               #
#######################################################################

from copy import copy
from random import choice

import numpy as np

from deepnotakto.agents import Agent as BaseAgent
from deepnotakto.agents import QTree as BaseQTree
from deepnotakto.agents import QTreeTrainer as BaseQTreeTrainer
from deepnotakto.environment import Env as BaseEnv
from deepnotakto.treesearch import GuidedNode as BaseGuidedNode
from deepnotakto.treesearch import Node as BaseNode
from deepnotakto.util import array_in_list, rotate, create_board


class Node (BaseNode):
    def action_space(self, state = None, remove_losses = True,
                     remove_isometries = True, get_probs = False):
        if state is None:
            state = self.state
        probs = []
        if get_probs:
            probs = self.action_space_probs(state)
        return _action_space(state = state,
                             player = self.player,
                             remove_losses = remove_losses,
                             remove_isometries = remove_isometries,
                             get_probs = get_probs,
                             probs = probs)

    def play_move(self, move, state = None):
        if state is None:
            state = copy(self.state)
        else:
            state = copy(state)
        state[move // state.shape[0], move % state.shape[0]] = 1
        return state

    def legal_move(self, move, state = None):
        """ Is a given move on a given state (or current state) legal? """
        if state is None:
            state = copy(self.state)
        else:
            state = copy(state)
        return state[move // state.shape[0], move % state.shape[0]] == 0

    def get_winner(self, state = None):
        """ Get the winner of the game """
        if state is None:
            state = copy(self.state)
        return _winner(state)

    def get_player(self):
        """ Calcualate current player by turn count """
        return 2 - int((np.sum(self.state) % 2))


class GuidedNode (Node, BaseGuidedNode):
    pass


class QTree (BaseQTree):
    def _node_defaults(self):
        """ Get the defaults for a new node """
        parent = super(QTree, self)._node_defaults()
        parent["state"] = np.zeros([self.size, self.size])
        parent["network"] = self
        return parent

    def _init_trainer(self, params = None, **kwargs):
        """ Initialize a trainer object """
        self.trainer = QTreeTrainer(self, params = params, **kwargs)

    def get_passable_state(self, state):
        """
        Prepare a feeding tensor for the computation graph of the model

        Args:
            state: (array or array[]) State or states to clean for network feed
        Returns:
            (array or array[]) State or states fit to be passed through network
        """
        # If multiple states are passed in
        if type(state) == list:
            # Add current player if needed to each state and flatten each state
            if self.player_as_input:
                feed = [np.concatenate((np.reshape(s, -1),
                                        [1] if np.sum(s) % 2 == 0 else [-1]))
                        for s in state]
            else:
                # Flatten each state
                feed = [np.reshape(s, -1) for s in state]
        # If only a single state passed in
        elif type(state) == np.ndarray:
            # Add current player if needed and flatten state
            if self.player_as_input:
                feed = [np.concatenate((np.reshape(state, -1),
                                        [1] if np.sum(state) % 2 == 0 else [
                                            -1]))]
            else:
                # Flatten state
                feed = [np.reshape(state, -1)]
        else:
            raise (ValueError("'state' must be a list of arrays or an array"))
        return feed

    def new_node(self, **kwargs):
        """ Create a new node with given parameters resolving to defaults """
        params = self._node_param_resolve(**kwargs)
        return GuidedNode(**params)

    def is_over(self, board):
        """ Checks if a game is over """
        if _winner(board) != 0:
            return True
        return False

    def action_space(self, state):
        """
        Returns a list of all possible moves (reguardless of win / loss)

        Args:
            state: (array) Current board state
        Returns:
            (array[]) - All legal moves for the given board
        """
        return [create_board(i, state.shape[0])
                for i in _action_space(state, remove_isometries = False,
                                       remove_losses = False,
                                       get_probs = False)]


class QTreeTrainer (BaseQTreeTrainer):
    def offline(self, states = None, policies = None, winners = None,
                batch_size = None, epochs = None, learn_rate = None,
                rotate = True):
        # Rotate if required
        if rotate and states is not None:
            states, policies, winners = self.get_rotations(states, policies,
                                                           winners)
        super(QTreeTrainer, self).offline(states, policies, winners,
                                          batch_size, epochs, learn_rate)

    def get_rotations(self, states, policies, winners):
        """ Add rotated copies of states and policies for a training set """
        # Copy states so as not to edit outside of scope
        states = list(states)
        policies = list(policies)
        # Collect rotated versions of each state and target
        new_states = []
        new_policies = []
        new_winners = []
        for s, p, w in zip(states, policies, winners):
            # Aliases for rotating and renaming
            s_ = s
            p_ = p
            for i in range(3):
                # Rotate them
                rs = rotate(s_)
                rp = rotate(p_)
                # Add them to the new lists
                new_states.append(rs)
                new_policies.append(rp)
                new_winners.append(w)
                # Rename the rotations to be normal
                s_ = rs
                p_ = rp
        # Combine lists
        return [states + new_states,
                policies + new_policies,
                winners + new_winners]


class Env (BaseEnv):
    def __init__(self, size, starting = None, rewards = None):
        """
        Initializes the environment
        Args:
            size: (int) Side length of the board
            starting: (array) Starting board configuration if not blank
            rewards: (dict) Custom reward values
        """
        self.size = size
        self.shape = (size, size)
        if starting is None:
            self.starting = np.zeros(self.shape, dtype = np.int8)
        else:
            self.starting = starting
        self.starting_turn = np.sum(self.starting)
        self.reset()
        if rewards is None:
            self.rewards = {
                "illegal": -10,
                "forced": 2,
                "loss": -2
            }
        else:
            self.rewards = rewards

    def illegal(self, state = None):
        if state is None:
            state = self.state
        if np.max(state) > 1:
            return True
        return False

    def play_move_on_state(self, state, action):
        return np.add(state, action)

    def forced(self, board = None):
        """Is a loss forced on the next turn"""
        if board is None:
            board = self.board
        return _forced(board)

    def reward(self, action):
        """
        Returns the immediate reward for a given action

        Args:
            action: (action) Action to play on game state
        Returns:
            (int) Reward for given action
        """
        # Play the move on a copy of the board
        new_state = self.play_move_on_state(self.state, action)
        # If illegal move, highly negative reward
        if self.illegal(new_state):
            return self.rewards["illegal"]

        # Rewards based on winner
        winner = self.winner(new_state)
        if winner == 0:
            # Positive reward for forcing a loss
            if self.forced(new_state):
                return self.rewards["forced"]
            # Otherwise, no reward
            return 0
        else:
            # Negative reward for a loss
            return self.rewards["loss"]

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
        # Get board
        if not isinstance(state, np.ndarray):
            state = copy(self.board)
        return [create_board(i, state.shape[0])
                for i in _action_space(state, remove_isometries = False,
                                       remove_losses = False,
                                       get_probs = False)]

    def __str__(self):
        """ Conversion to string """
        out = ""
        for i in self.state:
            for j in i:
                out += "O" if j == 0 else "X"
                out += " "
            out += "\n"
        return out


class RandomAgent (BaseAgent):
    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = "Random"

    def get_action(self, state):
        possible = _action_space(state)
        player = 1 if np.sum(state) % 2 == 0 else 2
        opponent = 2 if player == 1 else 1
        not_loser = []
        # Choose a winner and identify non-losers
        for m in possible:
            # Make move temporarily
            new_state = copy(state)
            new_state[m // state.shape[0], m % state.shape[0]] = 1
            winner = _winner(new_state)
            # If it is a loser, do not consider
            if winner == opponent:
                continue
            # If forced loss on opponent, choose move
            if _forced(new_state):
                return create_board(m, state.shape[0])
            # If neither, remember that it is a not a losing move
            not_loser.append(m)
        if len(not_loser) >= 1:
            # If there are non-losing move, choose one randomly
            return create_board(choice(not_loser), state.shape[0])
        else:
            # If all moves are losses, choose any
            return create_board(choice(possible), state.shape[0])


def measure(agent, **stats):
    """ Measure an agent and return the results along with any passed stats """
    # Zero board
    z = np.zeros(agent.shape)
    # Value of zero board
    stats["zero_val"] = agent.value(z)
    # Size of raw predictions of zero board
    stats["zero_norm"] = np.linalg.norm(agent.raw(z))
    # Maximum probability on zero board
    policy = agent.policy(z)
    stats["zero_max"] = np.max(policy)
    # Mean of probabilities on zero board
    stats["zero_mean"] = np.mean(policy)
    return stats


def _forced(board):
    """ Is a loss forced on the next turn """
    # Calculate possible moves for opponent
    remaining = _action_space(board)
    # If all are terminal, a loss is forced
    for r in remaining:
        if _winner(np.add(board, create_board(r, board.shape[0]))) == 0:
            return False
    return True


def _winner(state):
    state = copy(state)
    turn = np.sum(state)
    # Rows
    for row in state:
        if np.sum(row) == state.shape[0]:
            return 1 if turn % 2 == 0 else 2
    # Columns (row in transpose of b)
    for col in state.T:
        if np.sum(col) == state.shape[0]:
            return 1 if turn % 2 == 0 else 2
    # Diagonals
    # Top left to bottom right
    tlbr = np.sum(state * np.identity(state.shape[0]))
    if tlbr >= state.shape[0]:
        return 1 if turn % 2 == 0 else 2
    # Bottom left to top right
    bltr = np.sum(state * np.flip(np.identity(state.shape[0]), 1))
    if bltr >= state.shape[0]:
        return 1 if turn % 2 == 0 else 2
    # Otherwise game is not over
    return 0


def _action_space(state, player = 0, remove_losses = False,
                  remove_isometries = False, get_probs = False, probs = []):
    state = copy(state)
    if _winner(state) != 0:
        if get_probs:
            return [[], []]
        else:
            return []
    remaining = []
    remain_arrays = []
    # Get policies
    if get_probs:
        remain_probs = []
    # Loop over both axes
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            # If there is an empty space
            if state[i, j] == 0:
                # Play move
                nb = copy(state)
                nb[i, j] = 1
                # Remove losers if losses are to be removed
                winner = _winner(nb)
                winner_list = [0, 3 - player]
                if winner in winner_list if remove_losses else [0, 1, 2]:
                    if not remove_isometries:
                        # Add move to returning list
                        remaining.append((i * state.shape[0]) + j)
                        if get_probs:
                            remain_probs.append(probs[i, j])
                    else:
                        # Check if it is an isomorphism
                        if len(remain_arrays) > 0:
                            if array_in_list(nb, remain_arrays):
                                continue
                        # Add all isomorphisms to the remaining arrays
                        for _ in range(4):
                            if not array_in_list(nb, remain_arrays):
                                remain_arrays.append(nb)
                            if not array_in_list(nb.T, remain_arrays):
                                remain_arrays.append(nb.T)
                            nb = rotate(nb)
                        # Add move to returning list
                        remaining.append((i * state.shape[0]) + j)
                        if get_probs:
                            remain_probs.append(probs[i, j])
    if get_probs:
        return remaining, remain_probs
    return remaining
