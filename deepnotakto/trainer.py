# trainer.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from random import shuffle
from util import rotate as rotate_func

class Trainer (object):
    def __init__(self, agent, path = None, tensorboard_interval = 100):
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
        self.iteration = 0
        # If recording, setup tensorboard functions
        if tensorboard_interval >= 1:
            if path == None:
                path = "tensorboard/"
            self.writer = tf.summary.FileWriter(path + agent.name + "/")
            self.tensorboard_interval = tensorboard_interval
            self.record = True
        else:
            self.record = False

    def online(self, **kwargs):
        """Gets a callable function for online training"""
        return lambda s, a, r: self.online([s], [a], [r], 1, 1)

    def train(self, states, actions, rewards, batch_size = 1, epochs = 1):
        """
        Trains the agent over a set of state, action, reward triples
        Parameters:
            states (List of (N, N) arrays) - List of states
            actions (List of (N, N) arrays) - Actions taken on states
            rewards (List of floats) - Rewards for each action
            batch_size (int) - Number of samples in each minibatch
            epochs (int) - Number of iterations over the entire dataset
        """
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
        for i in range(0, len(l), n):
            yield l[i:i + n]