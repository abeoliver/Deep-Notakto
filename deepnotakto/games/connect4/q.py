import numpy as np
from deepnotakto import Node as BaseNode
from deepnotakto import GuidedNode as BaseGuidedNode
from deepnotakto import QTree as BaseQTree
from deepnotakto import QTreeTrainer
from deepnotakto.games.connect4 import game


class Node (BaseNode):
    def action_space(self, state = None, remove_losses = True,
                     get_probs = True):
        if state is None:
            state = self.state
        if not get_probs:
            return game.action_space(state, remove_losses)
        return [game.action_space(state, remove_losses),
                self.action_space_probs(state)]

    def play_move(self, move, state = None):
        if state is None:
            state = self.state
        game.play_move_on_state(state, move, game.current_player(state))

    def legal_move(self, move, state = None):
        """ Is a given move on a given state (or current state) legal? """
        if state is None:
            state = self.state
        return game.legal(state, move)

    def get_winner(self, state = None):
        """ Get the winner of the game """
        if state is None:
            state = self.state
        return game.winner(state)

    def get_player(self, state = None):
        """ Get current player number """
        if state is None:
            state = self.state
        return game.current_player(state)


class GuidedNode (Node, BaseGuidedNode):
    """ Inherits game rules from Node and AlphaZero fro, BaseGuidedNode """
    pass


class QTree (BaseQTree):
    def _node_defaults(self):
        """ Get the defaults for a new node """
        parent = super(QTree, self)._node_defaults()
        parent["state"] = np.zeros(self.size)
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
                feed = [np.concatenate((
                    np.reshape(s, -1),
                    [1] if np.sum(game.without_players(s)) % 2 == 0 else [-1]))
                        for s in state]
            else:
                # Flatten each state
                feed = [np.reshape(s, -1) for s in state]
        # If only a single state passed in
        elif type(state) == np.ndarray:
            # Add current player if needed and flatten state
            if self.player_as_input:
                feed = [np.concatenate((
                    np.reshape(state, -1),
                    [1] if np.sum(game.without_players(state)) % 2 == 0
                    else [-1]))]
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
        if game.winner(board) != 0:
            return True
        return False

    def action_space(self, state):
        """
        Returns a list of all possible moves (regardless of win / loss)

        Args:
            state: (array) Current board state
        Returns:
            (array[]) - All legal moves for the given board
        """
        return game.action_space(state, remove_losses = False)
