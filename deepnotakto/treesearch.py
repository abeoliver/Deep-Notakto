# treesearch.py
# Abraham Oliver, 2017
# Deep Notakto Project

# Inspired and originially structured by http://mcts.ai/code/python.html

from copy import copy
from random import choice, randrange
import numpy as np
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from deepnotakto.util import rotate, array_in_list
from deepnotakto.environment import Env
import deepnotakto.util as util

class Node (object):
    """
    A Node object

    Definitions:
        A 'State' is any type that represents the state of a game
        An 'Edge' is any type that tranfers State -> State

    Properties:
        state (State) - State that the node representd
        parent (Node or None) - Parent node
        edge (Edge) - Action from parent -> current
        children (Node[]) - Explored part of next state space
        visits (int) - Number of games played through this node
        wins (int) - Number of games won through this node
        unvisited (Edge[]) - Unexplored actions leading to next state space
        winner (int) - 0 if non-terminal, player number if terminal
        player (int) - Player number

    Methods:
        action_space : [None or State] -> Edge[]
            Possible edges from a given state
        get_winner : [None or State] -> int
            0 if non-terminal, player number if terminal
        play_move : Edge, [None or State] -> State
            Follows an edge to a new state from a state
        visit_unvisited : None -> Bool
            Chooses the next unvisited to visit and adds it to children
            Returns True if successful, returns False if no more to visit
        upadte : int -> None
            Updates a node with a given game result
        select : None -> Node
            Choose a child node to traverse
        get_player : None -> int
            Gets the number of a player with the node state
        random_move : None -> Node
            Gets a random node to move into
    """
    def __init__(self, state, parent = None, edge = None, visits = 0,
                 wins = 0, remove_unvisited_losses = True):
        """
        Creates a Node objedct
        Parameters:
            state (State) - State that the node representd
            parent (State) - State before action was taken
            edge (Edge) - Action from parent -> state
            children (Node[]) - Visited children nodes
            remove_unvisited_losses (bool) - Should losses be included in unvisited nodes
        """
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
        """ Remove this node from the tree and reset it"""
        self.parent = None
        self.edge = None
        self.children = []
        self.visits = 0
        self.wins = 0
        self.get_unvisited(remove_losses = self.remove_losses)

    def get_unvisited(self, remove_losses = False):
        self.unvisited = self.action_space(remove_losses = remove_losses)

    def __str__(self):
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
        if type(self.parent) == type(None):
            return "ROOT NODE after {}".format(self.visits)
        return "Node (P{}, W{}) (Ws {}, Vs {}) " \
               "State:  {}".format(
            str(self.player), str(self.winner), str(self.wins), str(self.visits),
            str(self.state).replace("\n", " "))

    def __getitem__(self, key):
        if type(key) != int:
            raise(KeyError("Key is not an integer"))
        elif abs(key) > len(self.children):
            raise(KeyError("Index out of range"))
        return self.children[key]

    def __iter__(self):
        for c in self.children:
            yield c

    def display_children(self):
        for i in self.children:
            print(i)
            print()

    def display(self):
        self.display_children()

    def get_player(self):
        return 0

    def action_space(self, state = None, remove_losses = False):
        return []

    def get_moves(self):
        return self.action_space()

    def random_move(self):
        return None

    def play_move(self, e, state = None):
        if type(state) == type(None):
            return self.state
        return state

    def get_winner(self, state = None):
        return 0

    def visit_unvisited(self, move = None):
        if len(self.unvisited) > 0:
            # Randomly choose a new move
            if move == None:
                move = choice(self.unvisited)
            # Remove move from untried moved
            self.unvisited.remove(move)
            # Create a new node for this child
            new_node = type(self)(state = self.play_move(move),
                                  parent = self,
                                  edge = move,
                                  remove_unvisited_losses = self.remove_losses)
            # Add new node to children
            self.children.append(new_node)
            return new_node
        else:
            raise(IndexError("Unvisited list is empty"))

    def update(self, winner):
        """ Updates a node with a loss (0) or a win (1) and traverses to its parent """
        self.visits += 1
        if winner == self.player:
            self.wins += 1
        if type(self.parent) != type(None):
            self.parent.update(winner)

    def best(self):
        if self.winner != 0:
            raise(Exception("Cannot select a child / move from a terminal node"))
        if  self.children == []:
            return self.visit_unvisited().edge
            # return choice(self.get_moves())
        return max(self.children, key = lambda c: c.visits).edge

    def ucb(self, c = None):
        if c == None:
            c = self
        return (c.wins / c.visits) + np.sqrt(2 * np.log(c.parent.visits) / c.visits)

    def select(self):
        """ Select the next child """
        return max(self.children, key = lambda c: self.ucb(c))

    def save(self, filename):
        with open(filename, "wb") as outFile:
            pickle_dump(self, outFile)

    def size(self):
        return len(self.children) + sum([i.size() for i in self.children])

    def get_child_by_state(self, state):
        for c in self.children:
            if c.state == state:
                return c

    def get_child_by_edge(self, edge):
        for c in self.children:
            if c.edge == edge:
                return c

    def get_policy(self, temperature = 1):
        """ Gets the policy for a given searched node """
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


class NotaktoNode (Node):
    def get_moves(self):
        return self.action_space(remove_losses = False, remove_isometries = False)

    def action_space(self, state = None, remove_losses = True, remove_isometries = True):
        if type(state) == type(None):
            state = self.state
        if self.get_winner(state) != 0:
            return []
        remaining = []
        remain_arrays = []
        # Loop over both axes
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                # If there is an empty space
                if state[i, j] == 0:
                    # Play move
                    nb = copy(state)
                    nb[i, j] = 1
                    # Check if it is not a loser (winner of played game should be other player)
                    # Include losses or not is defined by which list winner can be a part of
                    if self.get_winner(nb) in [0, 3 - self.player] if remove_losses else [0, 1, 2]:
                        if not remove_isometries:
                            # Add move to returning list
                            remaining.append((i * state.shape[0]) + j)
                        else:
                            # Check if it is an isomorphism
                            if len(remain_arrays) > 0:
                                if array_in_list(nb, remain_arrays):
                                    continue
                            # Add all isomorphisms to the remaining arrays (rotation and reflection)
                            for _ in range(4):
                                if not array_in_list(nb, remain_arrays):
                                    remain_arrays.append(nb)
                                if not array_in_list(nb.T, remain_arrays):
                                   remain_arrays.append(nb.T)
                                nb = rotate(nb)
                            # Add move to returning list
                            remaining.append((i * state.shape[0]) + j)
        return remaining

    def play_move(self, move, state = None):
        if type(state) == type(None):
            state = copy(self.state)
        else:
            state = copy(state)
        state[move // state.shape[0], move % state.shape[0]] = 1
        return state

    def legal_move(self, move, state = None):
        if type(state) == type(None):
            state = copy(self.state)
        else:
            state = copy(state)
        return state[move // state.shape[0], move % state.shape[0]] == 0

    def get_winner(self, board = None):
        if type(board) == type(None):
            board = copy(self.state)
        else:
            board = copy(board)
        # Rows
        for row in board:
            if np.sum(row) >= board.shape[0]:
                return int(1 + (np.sum(board) % 2))
        # Columns (row in transpose of b)
        for col in board.T:
            if np.sum(col) >= board.shape[0]:
                return int(1 + (np.sum(board) % 2))
        # Diagonals
        # Top left to bottom right
        tlbr = np.sum(board * np.identity(self.state.shape[0]))
        if tlbr >= self.state.shape[0]:
            return int(1 + (np.sum(board) % 2))
        # Bottom left to top right
        bltr = np.sum(board * np.flip(np.identity(self.state.shape[0]), 1))
        if bltr >= self.state.shape[0]:
            return int(1 + (np.sum(board) % 2))
        # Otherwise game is not over
        return 0

    def get_player(self):
        return 3 - int((np.sum(self.state) % 2) + 1)

    def random_move(self, remove_isometries = True, remove_losses = True):
        actions = self.action_space(remove_losses = remove_losses,
                                    remove_isometries = remove_isometries)
        if actions == []:
            return False
        action = choice(actions)
        node = NotaktoNode(self.play_move(action), self, action)
        return node

    def translate_move(self, target, move):
        """ Translates a move from the interal board to a target board """
        return translate_move(self.state, target, move)

    def child_with_isomorphic_move(self, move):
        """ Gets a child with a move isomorphic to a given one """
        new = self.play_move(move)
        for c in self.children:
            if isomorphic_matrix(c.state, new):
                return c

    def isomorphic_child(self, target):
        """ Find a child isomorpic to the target board"""
        # Check all children
        for c in self.children:
            if isomorphic_matrix(c.state, target):
                return c

    def get_child_state(self, state):
        for c in self.children:
            if np.array_equal(c.state, state):
                return c

    def forced(self, state = None):
        """Is a loss forced on the next turn"""
        if type(state) == type(None):
            s = copy(self.state)
        else:
            s = copy(state)
        # If (n-1)^2 + 1 pieces are played, then garaunteed force
        if np.sum(s) > (s.shape[0] - 1) ** 2:
            return True
        # Calculate possible moves for opponent
        remaining = self.action_space(s)
        # If all are losses, a loss is forced
        for r in remaining:
            if self.get_winner(self.play_move(r, s)) == 0:
                return False
        return True

    def choose_by_policy(self, policy = None, temperature = 1):
        if type(policy) == type(None):
            policy = self.get_policy(temperature)
        while True:
            action = np.random.choice(np.arange(self.state.size), p = policy)
            if not self.legal_move(action): continue
            if self.get_winner(self.play_move(action)) == 0:
                child = self.child_with_isomorphic_move(action)
                if child != None:
                    return child

    def choose_by_visits(self):
        return max(self.children, key = lambda c: c.visits)
        # Shouldn't need the rest of this
        cs = sorted(self.children, key = lambda c: c.visits, reverse = True)
        for child in cs:
            if not self.legal_move(child.edge): continue
            if child.winner in [0, child.player]:
                return child
        raise("No valid child")


class GuidedNotaktoNode (NotaktoNode):
    def __init__(self, state, network, parent = None, edge = None, visits = 0, wins = 0,
                 remove_unvisited_losses = True, total_value = 0, prior = 0, explore = 1):
        self.network = network
        self.unvisited_probs = []
        self.n = visits         # Visits to this state
        self.w = total_value    # Total value of this state
        if parent != None:
            self.p = prior      # Prior probability of this state
        else:
            self.p = 1.0        # If root node, prior is 100%
        self.explore = explore  # Exploration constant
        super(GuidedNotaktoNode, self).__init__(state = state, parent = parent, edge = edge,
                                                visits = visits, wins = wins,
                                                remove_unvisited_losses = remove_unvisited_losses)

    def separate(self):
        """ Remove this node from the tree and reset it"""
        self.unvisited_probs = []
        super(GuidedNotaktoNode, self).separate()
        self.n = 0
        self.w = 0
        self.p = 1.0

    def __str__(self):
        if type(self.parent) == type(None):
            return "ROOT NODE after {}".format(self.visits)
        return "Node (Player {}, Winner {})" \
               "\nVisits : {}, Total Value : {}, and Prior : {}" \
               "\nUpper Confidence Bound : {}" \
               "\nState:  {}".format(
            str(self.player), str(self.winner), str(self.n), str(self.w), str(self.p),
            str(self.ucb()), str(self.state).replace("\n", "\n\t")
        )

    def __repr__(self):
        if type(self.parent) == type(None):
            return "ROOT NODE after {} (P1 : {} wins)".format(self.visits, self.visits - self.wins)
        return "Node (P{}, W{}) (N {}, W {}, P {}) " \
               "State:  {}".format(
            str(self.player), str(self.winner), str(self.n), str(self.w), str(self.p),
            str(self.state).replace("\n", " "))

    def update(self, value):
        self.visits += 1
        self.n += 1
        self.w += value
        if type(self.parent) != type(None):
            self.parent.update(value)

    def ucb(self, c = None):
        if c == None:
            c = self
        return (c.w / c.n) + (c.explore * c.p *
                                    (np.sqrt(c.parent.n) / (1 + c.n)))

    def get_unvisited(self, remove_losses = True):
        self.unvisited, self.unvisited_probs = self.action_space(
            remove_losses = remove_losses, get_probs = True)

    def action_space(self, state = None, remove_losses = True, remove_isometries = True,
                     get_probs = False):
        if type(state) == type(None):
            state = self.state
        if self.get_winner(state) != 0:
            if get_probs: return [[], []]
            else: return []
        remaining = []
        remain_arrays = []
        # Get policies
        if get_probs:
            probs = self.network.policy(state).reshape(state.shape)
            remain_probs = []
        # Loop over both axes
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                # If there is an empty space
                if state[i, j] == 0:
                    # Play move
                    nb = copy(state)
                    nb[i, j] = 1
                    # Check if it is not a loser (winner of played game should be other player)
                    # Include losses or not is defined by which list winner can be a part of
                    if self.get_winner(nb) in [0, 3 - self.player] if remove_losses else [0, 1, 2]:
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
                            # Add all isomorphisms to the remaining arrays (rotation and reflection)
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
            return (remaining, remain_probs)
        return remaining

    def visit_unvisited(self, move = None):
        if len(self.unvisited) > 0:
            # Randomly choose a new move
            if move == None:
                move_index = randrange(len(self.unvisited))
            else:
                move_index = self.unvisited.index(move)
            move = self.unvisited.pop(move_index)
            prob = self.unvisited_probs.pop(move_index)
            # Create a new node for this child
            new_node = GuidedNotaktoNode(state = self.play_move(move),
                                         network = self.network,
                                         parent = self,
                                         edge = move,
                                         remove_unvisited_losses = self.remove_losses,
                                         prior = prob,
                                         explore = self.explore)
            # Add new node to children
            self.children.append(new_node)
            return new_node
        else:
            raise(IndexError("Unvisited list is empty"))

    def random_move(self, remove_isometries = True, remove_losses = True):
        actions = self.action_space(remove_losses = remove_losses,
                                    remove_isometries = remove_isometries)
        if actions == []:
            return False
        action = choice(actions)
        node = GuidedNotaktoNode(state = self.play_move(action),
                                 network = self.network,
                                 parent = self,
                                 edge = action,
                                 remove_unvisited_losses = self.remove_losses)
        return node

    @property
    def value(self):
        return self.network.value(self.state)


def load(filename, update = True):
    with open(filename, "rb") as f:
        if update:
            return update_version(pickle_load(f))
        return pickle_load(f)

def update_version(n, parent = None):
    new = NotaktoNode(n.state, parent, n.edge, visits = n.visits, wins = n.wins)
    new.unvisited = n.unvisited
    new.children = [update_version(i, new) for i in n.children]
    return new

def move_to_vec(move, size):
    x = np.zeros([size * size], dtype = np.int32)
    x[move] = 1
    return np.reshape(x, [size, size])

def rotate_move(move, size):
    return int(size * (size - 1 - (move % size)) + (move // size))

def rotate_move_cw(move, size):
    return int(size * (1 + move % size) - 1 - move // size)

def reflect_move(move, size):
    return int((move % size) * size + (move // size))

def translate_move(source, target, move):
    """
    Takes a move mapped on the source and translates it to the same relative
    move on the isomorphic target board
    """
    # Calculate type of isomorphosm
    for _ in range(4):
        # Identity
        if np.array_equal(target, source):
            return move
        # Reflection
        elif np.array_equal(target.T, source):
            return reflect_move(move, target.shape[0])
        # Rotate target
        target = rotate(target)
        # Rotate the move the other direction because it is based off the original board
        move = rotate_move_cw(move, target.shape[0])

def isomorphic_matrix(m1, m2):
    """ Checks if two matricies are isomorphic """
    # Check all isomorphisms
    for _ in range(4):
        if np.array_equal(m1, m2) or np.array_equal(m1.T, m2):
            return True
        # Rotate the target (rotates back to identity before it moves on)
        m1 = rotate(m1)
    return False

def search(root_node, iterations = 100):
    """ Run a MC tree search """
    # Run search 'iterations' times
    for i in range(iterations):
        # Start at the root node
        node = root_node

        # Selection phase (find a non-terminal / non-expanded node)
        # Traverse until an un-expanded node is found
        while node.unvisited == [] and node.children != []:
            # Move to the best child
            node = node.select()

        # Expansion Phase (choose an unvisited node to explore)
        # If non-terminal, choose an unvisited node
        if node.unvisited != []:
            node = node.visit_unvisited()

        # Rollout phase (run a random game from this node)
        while True:
            # Backpropagate winner
            if node.winner != 0:
                # The winner of the node wins
                node.update(node.winner)
                break
            elif node.action_space() == []:
                # Player that made this position wins
                node.update(node.player)
                break
            # Next node
            node = node.random_move()
    return root_node

def guidedsearch(root_node, iterations = 100):
    """ Run a MC tree search """
    # Run search 'iterations' times
    for i in range(iterations):
        # Start at the root node
        node = root_node
        # Selection phase (find a non-terminal / non-expanded node)
        # Traverse until an un-expanded node is found
        while node.unvisited == [] and node.children != []:
            # Move to the best child
            node = node.select()
        # Expansion phase
        # If there are some nodes to be visited, choose an unvisited node
        if node.unvisited != []:
            node = node.visit_unvisited()
        # Backup
        # Pass through the node's network value for its state
        node.update(node.value)
    return root_node