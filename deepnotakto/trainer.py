# trainer.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from random import shuffle, sample
from deepnotakto.util import rotate as rotate_func

class Trainer (object):
    def __init__(self, agent, training = None, path = None, tensorboard_interval = 0,
                 iterations = 0):
        """
        Initializes a Trainer object
        Parameter:
            agent (Agent) - Agent to train
            path (string) - File path to save Tensorboard info
                                (default "/tensorboard")
            tensorboard_interval (int) - Number of iterations between
                                            tensorboard saves
        Note:
            If tensorboard_interval is less than 1 then recording not implemented
        """
        self.agent = agent
        self.iteration = iterations
        self.training_params(training)
        # If recording, setup tensorboard functions
        if tensorboard_interval >= 1:
            if path == None:
                path = "tensorboard/"
            self.writer = tf.summary.FileWriter(path + agent.name + "/")
            self.tensorboard_interval = tensorboard_interval
            self.record = True
        else:
            self.record = False

    def change_param(self, name, value):
        self.params[name] = value

    def default_params(self):
        return {"replay_size": 1}

    def training_params(self, training = None):
        defaults = dict(self.default_params())
        if training == None:
            self.params = defaults
        else:
            self.params = training
            for key in defaults:
                if not key in self.params:
                    self.params[key] = defaults[key]

    def train(self, mode = None, source_agent = None, **kwargs):
        """Trains the model with the set training parameters"""
        options = {"online": self._online_mode, "episodic": self._episodic_mode,
                   "replay": self._replay_mode}
        if mode == None:
            options[self.params["type"]](source_agent, **kwargs)
        elif mode in options:
            options[mode](source_agent, **kwargs)

    def _online_mode(self, source_agent = None, **kwargs):
        if source_agent == None:
            source_agent = self.agent
        # Get move data
        ep = source_agent.episode[-1]
        # Run trainer
        self.online(ep[0], ep[1], ep[2], **kwargs)

    def _episodic_mode(self, source_agent = None, **kwargs):
        if source_agent == None:
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
        """Experience replay training mode"""
        if source_agent == None:
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
        """Gets a callable function for online training"""
        return lambda x: None

    def offline(self, **kwargs):
        pass

    def get_rotations(self, states, targets):
        """Train over the rotated versions of each state and reward"""
        # Copy states so as not to edit outside of scope
        states = list(states)
        # Collect rotated versions of each state and target
        new_states = []
        new_targets = []
        for s, t in zip(states, targets):
            ns = s
            nt = t
            for i in range(3):
                rs = rotate_func(ns)
                rt = rotate_func(nt)
                new_states.append(rs)
                new_targets.append(rt)
                ns = rs
                nt = rt
        # Combine lists
        all_states = states + new_states
        all_targets = targets + new_targets
        return [all_states, all_targets]

    def chunk(self, l, n):
        """
        Yield successive n-sized chunks from l
        Taken from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        """
        if n < 0:
            yield l
        for i in range(0, len(l), n):
            yield l[i:i + n]