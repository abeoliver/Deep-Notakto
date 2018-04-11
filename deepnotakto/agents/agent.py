#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: agents/agent.py                                              #
#  Abraham Oliver, 2018                                               #
#######################################################################

from collections import deque

from numpy import sum as np_sum
from numpy import zeros, identity, flip

from deepnotakto.trainer import Trainer


class Agent (object):
    def __init__(self, training = {"mode": None}, max_queue = 100):
        """
        Initialize an Agent object

        Args:
            training:  (dict) Training parameters used by a trainer
            max_queue: (int) Length of memory queue
        """
        self.states = deque(maxlen = max_queue)
        self.actions = deque(maxlen = max_queue)
        self.rewards = deque(maxlen = max_queue)
        self.episode = []
        self.max_queue = max_queue
        self.name = "agent"
        self.trainer = Trainer(self, training)

    def clear(self):
        """ Clear the memory queue and the epsiode """
        self.states = deque(maxlen = self.max_queue)
        self.actions = deque(maxlen = self.max_queue)
        self.rewards = deque(maxlen = self.max_queue)
        self.epsiode = []
    
    def act(self, env):
        """
        Choose action, apply action to environment, and recieve a reward

        Args:
            env: (Environment) Environment of the agent
        Returns:
            (Environment Observation) The output of an environment observation
        """
        # Current environment state
        state = env.observe()
        # Get the action
        action = self.get_action(state)
        # Apply action
        observation = env.act(action)
        # Record state, action, reward
        self.add_episode(state, action, observation["reward"])
        # Train online (may be avoided within the function w/ params params)
        if self.params["mode"] == "online":
            self.train("online")
        # Return the results
        return observation

    def get_action(self, state):
        """ Get the action to play on a given state """
        return None

    def train(self, mode = None, **kwargs):
        """ Trains an agent (needed for trainer API) """
        pass

    def new_episode(self):
        """ Reset the episode memory """
        self.episode = []

    def add_episode(self, state, action, reward):
        """ Adds a move of a game to a game episode """
        self.episode.append((state, action, reward))

    def save_episode(self, use_final = True):
        """
        Saves an episode to the game records

        Args:
            use_final: (bool) Save each move with the final reward of the game
        """
        # If episode hasn't been used, do nothing
        if len(self.episode) == 0:
            return None
        # If using final reward, record the final reward
        if use_final:
            # Fetch final reward
            final_reward = self.episode[-1][2]
            # Apply to each move
            self.episode = [(s, a, final_reward) for s, a, _ in self.episode]
        # Add move to memory queue
        for s, a, r in self.episode:
            self.states.append(s)
            self.actions.append(a)
            self.rewards.append(r)
        # Train episodically (according to training paramters)
        if self.params["mode"] in ["episodic", "replay"]:
            self.train(self.params["mode"])
        # Clear episode
        self.new_episode()

    def save(self, **kwargs):
        """ Saves an agent to a file """
        pass

    def change_training(self, **kwargs):
        """ Changes the training paramters """
        for key in kwargs:
            self.params[key] = kwargs[key]

    def get_q(self, state):
        """ Get the Q confidence value for a board (for reinforcement API) """
        return zeros(state.shape)

    def is_over(self, board):
        """ Checks if a game is over """
        return False

    @property
    def params(self):
        return self.trainer.params

    @params.setter
    def params(self, value):
        self.trainer.training_params(value)