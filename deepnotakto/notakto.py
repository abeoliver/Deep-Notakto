#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: notakto_treesearch.py                                        #
#  Abraham Oliver, 2018                                               #
#######################################################################

from copy import copy

import numpy as np

from deepnotakto.agents.qtree import QTree as BaseQTree
from deepnotakto.agents.qtree import QTreeTrainer as BaseQTreeTrainer
from deepnotakto.treesearch import GuidedNode as BaseGuidedNode
from deepnotakto.treesearch import Node as BaseNode
from deepnotakto.util import array_in_list, rotate


class Node (BaseNode):
    def action_space(self, state = None, remove_losses = True,
                     remove_isometries = True, get_probs = False):
        if state is None:
            state = self.state
        if self.get_winner(state) != 0:
            if get_probs:
                return [[], []]
            else:
                return []
        remaining = []
        remain_arrays = []
        # Get policies
        if get_probs:
            probs = self.action_space_probs(state)
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
                    winner = self.get_winner(nb)
                    winner_list = [0, 3 - self.player]
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
        else:
            state = copy(state)
        # Rows
        for row in state:
            if np.sum(row) >= state.shape[0]:
                return int(1 + (np.sum(state) % 2))
        # Columns (row in transpose of b)
        for col in state.T:
            if np.sum(col) >= state.shape[0]:
                return int(1 + (np.sum(state) % 2))
        # Diagonals
        # Top left to bottom right
        tlbr = np.sum(state * np.identity(self.state.shape[0]))
        if tlbr >= self.state.shape[0]:
            return int(1 + (np.sum(state) % 2))
        # Bottom left to top right
        bltr = np.sum(state * np.flip(np.identity(self.state.shape[0]), 1))
        if bltr >= self.state.shape[0]:
            return int(1 + (np.sum(state) % 2))
        # Otherwise game is not over
        return 0

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
        if self.guided:
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
