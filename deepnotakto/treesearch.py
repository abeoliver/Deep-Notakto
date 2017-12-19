# treesearch.py
# Abraham Oliver, 2017
# Deep Notakto Project

# Inspired and originially structured by http://mcts.ai/code/python.html

from copy import copy
from random import choice
import numpy as np

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
        UCB_select : None -> Node
            Choose a child node to traverse
        get_player : None -> int
            Gets the number of a player with the node state
        random_move : None -> Node
            Gets a random node to move into
    """
    def __init__(self, state, parent, edge):
        """
        Creates a Node objedct
        Parameters:
            state (State) - State that the node representd
            parent (State) - State before action was taken
            edge (Edge) - Action from parent -> state
            children (Node[]) - Visited children nodes
        """
        self.state = copy(state)
        self.parent = parent
        self.edge = edge
        self.children = []
        self.visits = 0
        self.wins = 0
        self.player = self.get_player()
        self.unvisited = self.action_space()
        self.winner = self.get_winner()

    def __repr__(self):
        return "Node" \
               "\nWins : {} and Visits : {}" \
               "\nState:  {}]".format(str(self.wins), str(self.visits), str(self.state).replace("\n", "\n\t"))

    def display_children(self):
        for i in self.children:
            print(i)
            print()

    def get_player(self):
        return 0

    def action_space(self, state = None):
        return []

    def random_move(self):
        return None

    def play_move(self, e, state = None):
        if type(state) == type(None):
            return self.state
        return state

    def get_winner(self, state = None):
        return 0

    def visit_unvisited(self, node_type = Node):
        if len(self.unvisited) > 0:
            # Randomly choose a new move
            move = choice(self.unvisited)
            # Remove move from untried moved
            self.unvisited.remove(move)
            # Create a new node for this child
            new_node = node_type(self.play_move(move), self, move)
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

    def UCB_select(self):
        """ Select the next child """
        return sorted(self.children, key = lambda c: c.wins / c.visits + np.sqrt(2 * np.log(self.visits) / c.visits))[-1]

class NotaktoNode (Node):
    def action_space(self, state = None):
        if type(state) == type(None):
            state = self.state
        remaining = []
        # Loop over both axes
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                # If there is an empty space
                if state[i, j] == 0:
                    # If it is not a loss
                    nb = copy(state)
                    nb[i, j] = 1
                    if self.get_winner(nb) in [0, self.player]:
                        remaining.append((i * state.shape[0]) + j)
        return remaining

    def play_move(self, move, state = None):
        if type(state) == type(None):
            state = copy(self.state)
        else:
            state = copy(state)
        state[move // state.shape[0], move % state.shape[0]] = 1
        return state

    def get_winner(self, board = None):
        if type(board) == type(None):
            board = copy(self.state)
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
        return int((np.sum(self.state) % 2) + 1)

    def random_move(self):
        action = choice(self.action_space())
        node = NotaktoNode(self.play_move(action), self, action)
        return node

    def visit_unvisited(self):
        return super(NotaktoNode, self).visit_unvisited(NotaktoNode)

def search(size, iterations = 100):
    """ Run a MC tree search """
    # Create root node
    root_node = NotaktoNode(np.zeros([size, size], dtype = np.int32), None, None)
    # Run search 'iterations' times
    for i in range(iterations):
        #print(len(root_node.children))
        # Start at the root node
        node = root_node
        # Iteration end signal
        end = False

        # Selection phase (find a non-terminal / non-expanded node)
        # Traverse until an un-expanded node is found
        while node.unvisited == [] and node.children != []:
            # Move to the best child
            node = node.UCB_select()
            # If this is a terminal node, update and end
            if node.winner != 0:
                end = True
                node.update(node.winner)
                break
        # End this iteration
        if end: continue

        # Expansion Phase (choose an unvisited node to explore)
        # If non-terminal, choose an unvisited node
        if node.unvisited != []:
            node = node.visit_unvisited()
        # If terminal, update
        elif node.winner != 0:
            end = True
            node.update(node.winner)
            break
        else:
            break

        # Rollout phase (run a random game from this node)
        while True:
            # Backpropagate winner
            if node.winner != 0:
                node.update(node.winner)
                break
            if node.action_space() == []:
                node.update(node.player)
                break
            # Next node
            node = node.random_move()
    return root_node