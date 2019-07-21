#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: treesearch.py                                                #
#  Abraham Oliver, 2018                                               #
#######################################################################

# Inspired and originially structured by http://mcts.ai/code/python.html
# Guided search modeled after Silver et al's AlphaZero algorithm (2017)

from copy import copy
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from random import choice, randrange

import numpy as np

import deepnotakto.util as util
from deepnotakto.util import rotate, array_in_list


class Node (object):
    """
    A Node object

    Definitions:
        A 'State' is any type that represents the state of a game
        An 'Edge' is any type that tranfers State -> State

    Properties:
        state (State) - State that the node represents
        parent (Node or None) - Parent node
        edge (Edge or None) - Previous action from parent -> current
        children (Node[]) - Explored part of next state space
        visits (int) - Number of games played through this node
        wins (int) - Number of games won through this node
        unvisited (Edge[]) - Unexplored actions leading to next state space
        winner (int) - 0 if non-terminal, player number if terminal
        player (int) - Player number
    """
    def __init__(self, state, parent = None, edge = None, visits = 0,
                 wins = 0, remove_unvisited_losses = True):
        """
        Creates a Node object
        Args:
            state: (State) State that the node representd
            parent: (State) State before action was taken
            edge: (Edge) Action from parent -> state
            children: (Node[]) Visited children nodes
            remove_unvisited_losses: (bool) Should visit losing nodes?
        """
        self._type = "normal"
        self.state = copy(state)
        self.parent = parent
        self.edge = edge
        self.children = []
        self.visits = visits
        self.wins = wins
        self.player = self.get_player()
        self.remove_losses = remove_unvisited_losses
        self.get_unvisited(remove_losses = remove_unvisited_losses)
        self.winner = self.get_winner()

    def separate(self):
        """ Remove this node from the tree and reset it """
        self.parent = None
        self.edge = None
        self.children = []
        self.visits = 0
        self.wins = 0
        self.get_unvisited(remove_losses = self.remove_losses)

    def get_unvisited(self, remove_losses = True):
        self.unvisited, self.unvisited_probs = self.action_space(
            remove_losses = remove_losses, get_probs = True
        )

    def __str__(self):
        """ Allow for pretty printing of a node """
        if type(self.parent) == type(None):
            return "ROOT NODE after {}".format(self.visits)
        return "Node (Player {}, Winner {})" \
               "\nWins : {} and Visits : {}" \
               "\nUpper Confidence Bound : {}" \
               "\nState:  {}".format(
            str(self.player), str(self.winner), str(self.wins), str(self.visits),
            str(self.ucb()), str(self.state).replace("\n", "\n\t")
        )

    def __repr__(self):
        """ Allow for less-than-pretty printing """
        if type(self.parent) == type(None):
            return "ROOT NODE after {}".format(self.visits)
        return "Node (P{}, W{}) (Ws {}, Vs {}) " \
               "State:  {}".format(
            str(self.player), str(self.winner), str(self.wins), str(self.visits),
            str(self.state).replace("\n", " "))

    def __getitem__(self, key):
        """ Allow for indexing of children """
        if type(key) != int:
            raise(KeyError("Key is not an integer"))
        elif abs(key) > len(self.children):
            raise(KeyError("Index out of range"))
        return self.children[key]

    def __iter__(self):
        """ Allow for iterating over children """
        for c in self.children:
            yield c

    def display_children(self):
        """ Pretty print all children """
        for i in self.children:
            print(str(i) + "\n")

    def get_player(self):
        """ Get the current player at the node """
        return self.player

    def action_space(self, state = None, remove_losses = False,
                     get_probs = True):
        """ Get all available actions and (if requested) their probabilities """
        if not get_probs:
            return []
        return [[], []]

    def random_move(self, remove_losses = True):
        """ Play a random move """
        actions = self.action_space(remove_losses = remove_losses,
                                    get_probs = False)
        if not actions:
            return False
        action = choice(actions)
        node = self.__class__(state = self.play_move(action),
                              parent = self,
                              edge = action,
                              remove_unvisited_losses = self.remove_losses)
        return node

    def play_move(self, move, state = None):
        """ Play a move on the given state """
        if state is None:
            return self.state
        return state

    def get_winner(self, state = None):
        """ Get the winner of a given state """
        return 0

    def visit_unvisited(self, move = None):
        """ Visit an unvisited node either by a given move or by random move """
        if len(self.unvisited) > 0:
            # Randomly choose a new move
            if move is None:
                move = choice(self.unvisited)
            # Remove move from untried moved
            self.unvisited.remove(move)
            # Create a new node for this child
            new_node = self.__class__(
                state = self.play_move(move), parent = self, edge = move,
                remove_unvisited_losses = self.remove_losses)
            # Add new node to children
            self.children.append(new_node)
            return new_node
        else:
            raise(IndexError("Unvisited list is empty"))

    def update(self, winner):
        """ Updates a node with game outcome and traverses to its parent """
        self.visits += 1
        self.wins += 1 if winner == self.player else 0
        if type(self.parent) != type(None):
            # Switch winner for perspective of other player
            self.parent.update(-1 * winner)

    def best(self):
        """ Choose the best move by number of visits"""
        if self.winner != 0:
            raise(Exception("Cannot select a child / move from a terminal node"))
        if  self.children == []:
            return self.visit_unvisited().edge
        return max(self.children, key = lambda c: c.visits).edge

    def legal_move(self, move):
        """ Decide if a move is legal """
        return True

    def ucb(self, c = None):
        """ Calculate the upper confidence bound for either the node or a given node """
        if c == None:
            c = self
        if c.visits == 0:
            return 0.0
        return (c.wins / c.visits) + np.sqrt(2 * np.log(c.parent.visits) / c.visits)

    def select(self):
        """ Select the next child based on UCB score """
        return max(self.children, key = lambda c: self.ucb(c))

    def save(self, filename):
        """ Save a node to a file by filename """
        with open(filename, "wb") as outFile:
            pickle_dump(self, outFile)

    def size(self):
        """ Calculate the size of a given tree"""
        return len(self.children) + sum([i.size() for i in self.children])

    def get_child_state(self, state):
        """ Get a child with a given state"""
        for c in self.children:
            if np.array_equal(c.state, state):
                return c

    def get_child_by_edge(self, edge):
        """ Find a child node with a given edge and return it"""
        for c in self.children:
            if c.edge == edge:
                return c

    def get_policy(self, temperature = 1):
        """ Calculates the policy for a given searched node """
        # High temperature -> More exploration
        visits = []
        for action in range(self.state.size):
            # Get child with this action
            found = False
            for c in self.children:
                if c.edge == action:
                    visits.append(c.visits)
                    found = True
                    break
            # Default
            if not found:
                visits.append(0)
        visits = np.array(visits)
        # return util.softmax(np.power(visits, 1 / temperature))
        raised = np.power(visits, 1 / temperature)
        summed = np.sum(raised)
        if summed == 0.0:
            return np.ones(raised.shape) / raised.size
        return raised / summed

    def choose_by_visits(self):
        """ Choose the next child based on visit count """
        s = sorted(self.children, key = lambda c: c.visits, reverse = True)
        if len(s) >= 2:
            # If multiple with same count
            if s[0].visits == s[1].visits:
                # Get all with same visit count
                f = filter(lambda c: c.visits == s[0].visits, s)
                # Get maximum wins from these
                return max(f, key = lambda c: c.wins)
        return s[0]

    def action_space_probs(self, state):
        return util.softmax(np.zeros(state.shape))

    def choose_by_policy(self, policy = None, temperature = 1):
        """
        Choose a move based on a policy

        Args:
            policy: (Nd-array or None) Probability distribution over moves
            temperature: (float) Temperature hyperparameter
        Returns:
            (Node) The choosen node from the policy
         """
        # Get the current network policy if none is given
        if policy is None:
            policy = self.get_policy(temperature)
        # Arrange acions randomly but weighted by the policy
        actions = np.random.choice(self.state.size, np.count_nonzero(policy),
                                   replace = False, p = policy)
        # Find a non-illegal move in the order arranged
        for move in actions:
            if self.legal_move(move):
                # Create a node with the choosen move
                return self.__class__(state = self.play_move(move),
                                      parent = self,
                                      edge = move)
        # Raise an error if there are no legal moves to play
        raise "No legal actions to choose from"


class GuidedNode (Node):
    def __init__(self, state, network = None, parent = None, edge = None,
                 visits = 0, remove_unvisited_losses = True, total_value = 0,
                 prior = 0, explore = 1):
        # Get the network for the parent if none is given
        if network is None:
            if parent is None:
                raise "A network must be given or passed down by a parent"
            self.network = parent.network
        else:
            self.network = network
        self.unvisited_probs = []
        self.n = visits         # Visits to this state
        self.w = total_value    # Total value of this state
        if parent is None:
            self.p = prior      # Prior probability of this state
        else:
            self.p = 1.0        # If root node, prior is 100%
        self.explore = explore  # Exploration constant
        super(GuidedNode, self).__init__(
            state = state, parent = parent, edge = edge, visits = visits,
            remove_unvisited_losses = remove_unvisited_losses)
        self._type = "guided"

    def separate(self):
        """ Remove this node from the tree and reset it"""
        self.unvisited_probs = []
        super(GuidedNode, self).separate()
        self.n = 0
        self.w = 0
        self.p = 1.0

    def __str__(self):
        if self.parent is None:
            return "ROOT NODE (Visits : {}, Value: {})".format(self.n, self.w)
        return "Node (Player {}, Winner {})" \
               "\nVisits : {}, Total Value : {}, and Prior : {}" \
               "\nUpper Confidence Bound : {}" \
               "\nState:  {}".format(str(self.player), str(self.winner),
                                     str(self.n), str(self.w), str(self.p),
                                     str(self.ucb()),
                                     str(self.state).replace("\n", "\n\t")
        )

    def __repr__(self):
        if self.parent is None:
            return "ROOT NODE (N : {}, W : {})".format(self.n, self.w)
        return "Node (P{}, W{}) (N {}, W {}, P {}) " \
               "State:  {}".format(str(self.player), str(self.winner),
                                   str(self.n), str(self.w), str(self.p),
                                   str(self.state).replace("\n", " "))

    def update(self, value):
        self.visits += 1
        self.n += 1
        self.w += value
        if self.parent is not None:
            self.parent.update(-1 * value)

    def ucb(self, c = None):
        if c is None:
            c = self
        if c.n == 0:
            return 0.0
        # return c.p * ((c.w / c.n) + (c.explore * (np.sqrt(c.parent.n) /
        # (1 + c.n))))
        # return (c.p * c.w / c.n) + (c.explore * (np.sqrt(c.parent.n) /
        # (1 + c.n)))
        return (c.w / c.n) + (c.explore * c.p * np.sqrt(c.parent.n) / (1 + c.n))

    def visit_unvisited(self, move = None):
        if len(self.unvisited) > 0:
            # Randomly choose a new move
            if move is None:
                move_index = randrange(len(self.unvisited))
            else:
                move_index = self.unvisited.index(move)
            move = self.unvisited.pop(move_index)
            prob = self.unvisited_probs.pop(move_index)
            # Create a new node for this child
            r_loss = self.remove_losses
            new_node = self.__class__(state = self.play_move(move),
                                      network = self.network,
                                      parent = self,
                                      edge = move,
                                      remove_unvisited_losses = r_loss,
                                      prior = prob,
                                      explore = self.explore)
            # Add new node to children
            self.children.append(new_node)
            return new_node
        else:
            raise(IndexError("Unvisited list is empty"))

    def action_space_probs(self, state):
        return self.network.get_q(state)

    def choose_by_visits(self):
        s = sorted(self.children, key = lambda c: c.n, reverse = True)
        if len(s) >= 2:
            # If multiple with same count
            if s[0].visits == s[1].visits:
                # Get all with same visit count
                f = filter(lambda c: c.n == s[0].n, s)
                # Get maximum action value from these
                return max(f, key = lambda c: c.w)
        return s[0]

    @property
    def value(self):
        return self.network.value(self.state)


def load(filename):
    with open(filename, "rb") as f:
        return pickle_load(f)


def search(root_node, simulations = 100, modified = True):
    """
    Run a monte-carlo tree search from a node

    Note: a modified search uses the value of the node to determine the
    estimation of winning instead of rollouts consistent with Silver et al

    Args:
        root_node: (Node) Root node to start search from
        simulations: (int) Number of simulations to run search for
        modified: (bool) Should skip the search for value evaluation
    Returns:
        (Node) A searched expansion of the original root node
    """
    # Do not run modified search if the node is not a modified node
    if root_node._type != "guided":
        modified = False
    # Run search 'simulations' times
    for i in range(simulations):
        # Start at the root node
        node = root_node

        # End search if there is only one possible child
        if not node.unvisited and len(node.children) == 1:
            break

        # Selection phase (find a non-terminal / non-expanded node)
        # Traverse until an un-expanded node is found
        while not node.unvisited and node.children:
            # Move to the best child
            node = node.select()

        # Expansion Phase
        # Choose an unvisited node to explore if not fully expanded
        if node.unvisited:
            node = node.visit_unvisited()

        # If not modified, rollout phase
        # If modified, evaluate with model and pass value up network
        while True:
            # Backpropagate winner
            # If node is forced
            if len(node.action_space()) == 0:
                # Player that made this position wins
                node.update(1)
                break
            # If node is a winner (rarely)
            if node.winner != 0:
                # The winner of the node wins
                node.update(1 if node.winner == node.player else -1)
                break

            # If node is a modified node, do not rollout
            if modified:
                node.update(node.value)
                break
            else:
                # Next node
                node = node.random_move()
    return root_node

