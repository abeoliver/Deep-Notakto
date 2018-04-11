#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: trainer.py                                                   #
#  Abraham Oliver, 2018                                               #
#######################################################################

from random import sample

import tensorflow as tf

from deepnotakto.util import rotate as rotate_func


class Trainer (object):
    def __init__(self, agent, params = None, tensorboard_path = None,
                 tensorboard_interval = 0, iterations = 0):
        """
        Initializes a Trainer object

         Note:
            If tensorboard_interval is less than 1 then recording not used

        Args:
            agent: (Agent) Agent to train
            params: (dict) params parameters
            tensorboard_path: (string) File path to save Tensorboard info
                                        (default "/tensorboard")
            tensorboard_interval: (int) Iterations between tensorboard saves
        """
        self.agent = agent
        self.iteration = iterations
        self.training_params(params)
        # If recording, setup tensorboard functions
        self.tensorboard_interval = tensorboard_interval
        if tensorboard_interval >= 1:
            if tensorboard_path == None:
                tensorboard_path = "tensorboard/"
            self.tensorboard_path = tensorboard_path
            self.writer = tf.summary.FileWriter(
                tensorboard_path + agent.name + "/")
            self.tensorboard_interval = tensorboard_interval
            self.record = True
        else:
            self.record = False
            self.tensorboard_path = None

    def change_param(self, name, value):
        """ Change the value of a single training parameter """
        self.params[name] = value

    def default_params(self):
        """ Get the default training parameters """
        return {"replay_size": 1}

    def training_params(self, params = None):
        """ Resolve given set of parameters using defaults for missing items """
        defaults = dict(self.default_params())
        if params is None:
            self.params = defaults
        else:
            self.params = params
            for key in defaults:
                if key not in self.params:
                    self.params[key] = defaults[key]

    def train(self, mode = None, source_agent = None, **kwargs):
        """ Trains the model with the set of training parameters """
        options = {"episodic": self._episodic_mode, "replay": self._replay_mode}
        if mode is None:
            options[self.params["type"]](source_agent, **kwargs)
        elif mode in options:
            options[mode](source_agent, **kwargs)

    def _episodic_mode(self, source_agent = None, **kwargs):
        """ Train a model over a source agent's current episode """
        if source_agent is None:
            source_agent = self.agent
        # Get episode data
        states = []
        actions = []
        rewards = []
        for s, a, r in source_agent.episode:
            states.append(s)
            actions.append(a)
            rewards.append(r)
        # Run trainer
        self.offline(states, actions, rewards, **kwargs)

    def _replay_mode(self, source_agent = None, **kwargs):
        """ Train over an experience replay """
        if source_agent is None:
            source_agent = self.agent
        size = len(source_agent.states)
        if size > self.params["replay_size"]:
            indexes = sample(range(size), self.params["replay_size"])
        else:
            indexes = range(size)
        self.offline([source_agent.states[i] for i in indexes],
                     [source_agent.actions[i] for i in indexes],
                     [source_agent.rewards[i] for i in indexes],
                     **kwargs)

    def online(self, state, action, reward, **kwargs):
        """Gets a callable function for online params"""
        return lambda x: None

    def offline(self, states = None, actions = None, rewards = None,
                batch_size = None, epochs = None, learn_rate = None):
        pass
