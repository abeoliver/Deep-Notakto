# trainer.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np

class Trainer (object):
    def __init__(self, agent):
        """
        Initializes a Trainer object
        Parameter:
            agent (Agent.Agent) - Agent to train
        Returns:
            VOID
        """
        self.agent = agent
        self.iteration = 0

    def online(self, env, a2, train_a2 = False):
        """
        Train the agent in an environment against an agent
        Parameters:
            env (Environment.Env) - Environment to train in
            a2 (Agent.Agent) - Agent to train against
            train_a2 (bool) - Should train the other agent?
        Returns:
            VOID
        """
        pass

    def offline(self, states, actions, rewards):
        """
        Trains the agent with a Markov Decision Model
        Parameters:
            states (List of (N, N) arrays) - List of states
            actions (List of (N, N) arrays) - Actions taken on states
            rewards (List of floats) - Rewards for each action
        Returns:
            VOID
        """
        pass

    def batch(self, states, targets):
        """
        Trains the agent over a batch of states and targets
        Parameters:
            states (List of (N, N) arrays) - List of states
            targets (List of (N, N) arrays) - List of targets for each state
        Returns:
            VOID
        """
        pass