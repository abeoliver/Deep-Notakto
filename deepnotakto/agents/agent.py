# agent.py
# Abraham Oliver, 2017
# Deep-Notakto Project
from numpy.random import normal
from numpy import zeros, identity, flip
from numpy import sum as np_sum
from collections import deque

class Agent (object):
    def __init__(self, training = {"mode": None}, max_queue = 100):
        self.states = deque(maxlen = max_queue)
        self.actions = deque(maxlen = max_queue)
        self.rewards = deque(maxlen = max_queue)
        self.episode = []
        self.max_queue = max_queue
        self.training = training
        self.architecture = "N/A"
        self.name = "agent"
    
    def act(self, env):
        """
        Choose action, apply action to environment, and recieve reward
        Parameters:
            env (environment.Env) - Environment of the agent
        """
        # Current environment state
        state = env.observe()
        # Get the action
        action = self.get_action(state)
        # Apply action
        observation = env.act(action)
        # Record state, action, reward
        self.add_episode(state, action, observation["reward"])
        # Train online (may be avoided within the function w/ training params)
        if self.training["mode"] == "online":
            self.train("online")
        # Return the results
        return observation

    def train(self, mode = None, **kwargs):
        pass

    def new_episode(self):
        """Reset the memory of an agent for a new episode"""
        self.episode = []

    def add_episode(self, state, action, reward):
        """Adds a move of a game to a game episode"""
        self.episode.append((state, action, reward))

    def save_episode(self, use_final = False):
        """
        Saves an episode to the game records
        Parameters:
            use_final (bool) - Save the episode with the final reward for
                                    each reward in episode
            reward (float) - If not using final, other reward to use for
                                    each reward in episode
        """
        # If episode hasn't been used, do nothing
        if len(self.episode) == 0:
            return None
        # If using final reward, record the final reward
        if use_final:
            # Fetch final reward
            final_reward = self.episode[-1][2]
            if final_reward >= -1:
                self.episode = [(s, a, final_reward) for s, a, _ in self.episode]
        # Loop through each item in episode
        for s, a, r in self.episode:
            # Add to game record
            self.states.append(s)
            self.actions.append(a)
            self.rewards.append(r)
        # Train episodically (may be avoided within the function w/ training params)
        if self.training["mode"] in ["episodic", "replay"]:
            self.train(self.training["mode"])
        # Clear episode
        self.episode = []

    def save(self, **kwargs):
        """Saves an agent to a file"""
        pass

    def change_training(self, **kwargs):
        """Changes the training parameters"""
        for key in kwargs:
            self.training[key] = kwargs[key]

    def get_Q(self, state):
        return zeros(state.shape)

    def is_over(self, board):
        """Checks if game is over"""
        # Rows
        for row in board:
            if np_sum(row) == board.shape[0]:
                return True
        # Columns (row in transpose of b)
        for col in board.T:
            if np_sum(col) == board.shape[0]:
                return True
        # Diagonals
        # Top left to bottom right
        if np_sum(board * identity(self.size)) >= self.size:
            return True
        # Bottom left to top right
        if np_sum(board * flip(identity(self.size), 1)) >= self.size:
            return True
        # Otherwise game is not over
        return False