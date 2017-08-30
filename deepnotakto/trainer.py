# trainer.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from random import shuffle
from util import rotate as rotate_func

class Trainer (object):
    def __init__(self, agent, learn_rate = 1e-4, record = True,
                 change_agent_epsilon = False, epsilon_func = None,
                 path = None, tensorboard_interval = 100):
        """
        Initializes a Trainer object
        Parameter:
            agent (Q.Q) - Q Agent to train
            learn_rate (float) - Learning rate
            record (bool) - Record tensorboard data?
        """
        self.agent = agent
        self.iteration = 0
        if path == None:
            path = "tensorboard/"
        self.writer = tf.summary.FileWriter(path + agent.name + "/")
        self.tensorboard_interval = tensorboard_interval
        self.learn_rate = learn_rate
        self.record = record
        self.change_agent_epsilon = change_agent_epsilon
        if epsilon_func == None:
            self.epsilon_func = lambda x: 1.0 / (x + 1)
        else:
            self.epsilon_func = epsilon_func

    def get_online(self, learn_rate = None, **kwargs):
        """Gets a callable function of online with a given learning rate"""
        if learn_rate == None:
            learn_rate = self.learn_rate
        f = lambda s, a, r: self.online(s, a, r, learn_rate, **kwargs)
        f._type = "online"
        return f

    def online(self, state, action, reward, learn_rate = None,
               record = None, rotate = False, **kwargs):
        """
        Train the on a single state and reward (usually in an environment)
        Parameters:
            state ((N, N) array) - State
            action ((N, N) array) - Action applied
            reward (float) - Reward for action on state
            learn_rate (float) - Learning rate
            record (bool) - Record tensorboard info? (instance default if none)
        """
        if learn_rate == None:
            learn_rate = self.learn_rate
        if record == None:
            record = self.record
        target = self.agent.target(state, action,
                                   self.agent.get_Q(state), reward, **kwargs)
        if rotate:
            states, targets = self.get_rotations([state], [target])
        else:
            states, targets = [state], [target]
        summary = self.agent.update(states, targets, learn_rate)
        if record and (self.iteration % self.tensorboard_interval == 0):
            self.writer.add_summary(summary, self.iteration)
        self.iteration += 1
        if self.change_agent_epsilon:
            self.change_epsilon()

    def get_episode(self, learn_rate = None, rotate = False, **kwargs):
        """Gets a callable function of episode with a given learning rate"""
        if learn_rate == None:
            learn_rate = self.learn_rate
        f = lambda s, a, r: self.episode(s, a, r, learn_rate, rotate, **kwargs)
        f._type = "episode"
        return f

    def episode(self, states, actions, rewards, learn_rate = None, rotate = False,
                record = None, **kwargs):
        if learn_rate == None:
            learn_rate = self.learn_rate
        if record == None:
            record = self.record
        targets = [self.agent.target(state, action, self.agent.get_Q(state),
                                     reward, **kwargs)
                   for state, action, reward in
                   zip(states, actions, rewards)]
        if rotate:
            states, targets = self.get_rotations(states, targets)
        summary = self.agent.update(states, targets, learn_rate)
        if record and (self.iteration % self.tensorboard_interval == 0):
            self.writer.add_summary(summary, self.iteration)
        self.iteration += 1
        if self.change_agent_epsilon:
            self.change_epsilon()

    def offline(self, states, actions, rewards, batch_size = 1, epochs = 1,
                learn_rate = None, silence = False, record = None, rotate = False):
        """
        Trains the agent with a Markov Decision Model
        Parameters:
            states (List of (N, N) arrays) - List of states
            actions (List of (N, N) arrays) - Actions taken on states
            rewards (List of floats) - Rewards for each action
            batch_size (int) - Number of samples in each minibatch
            epochs (int) - Number of iterations over the entire dataset
            learn_rate (float) - Learning rate
            silence (bool) - Print to stdout?
            record (bool) - Record tensorboard info? (instance default if none)
        Note:
            Targets are caculated at the beginning of each epoch.
            Therefore, all targets in a given epochs use the same
            Q-function and are not effected by the others in the
            batch
        """
        if learn_rate == None:
            learn_rate = self.learn_rate
        # Output
        if not silence:
            print("Training ", end = "")
            display_interval = epochs // 10 if epochs >= 10 else 1
        for epoch in range(epochs):
            # Calculate targets from states, actions, and rewards
            targets = [self.agent.target(state, action,
                                          self.agent.get_Q(state), reward)
                       for (state, action, reward) in \
                       zip(states, actions, rewards)]
            if rotate:
                states, targets = self.get_rotations(states, targets)
            # Batch train once
            self.batch(states, targets, batch_size, 1, learn_rate, True, record)
            # Output
            if not silence:
                if epoch % display_interval == 0:
                    print("*", end = "")
        if not silence:
            print(" Done")

    def batch(self, states, targets, batch_size, epochs = 1,
              learn_rate = None, silence = False, record = None):
        """
        Trains the agent over a batch of states and targets
        Parameters:
            states (List of (N, N) arrays) - List of states
            targets (List of (N, N) arrays) - List of targets for each state
            batch_size (int) - Number of samples in each minibatch
            epochs (int) - Number of iterations over the entire dataset
            learn_rate (float) - Learning rate
            silence (bool) - Print to stdout?
            record (bool) - Record tensorboard info? (instance default if none)
        """
        if learn_rate == None:
            learn_rate = self.learn_rate
        if record == None:
            record = self.record
        # Output
        if not silence:
            print("Training ", end="")
            eb = (epochs * batch_size)
            display_interval = eb // 10 if eb >= 10 else 1
            display_iterations = 0
        # Train
        for epoch in range(epochs):
            # Batching ( get a list of batches of indicies)
            # Shuffle all indicies
            order = list(range(len(states)))
            shuffle(order)
            # Chunk into batches
            batches = list(self._chunk(order, batch_size))
            # Train over each minibatch
            for batch in batches:
                # Get the states and targets from the indidicies of the
                # batch and pass into update
                summary = self.agent.update([states[b] for b in batch],
                                             [targets[b] for b in batch],
                                             learn_rate)
                if record and (self.iteration % self.tensorboard_interval == 0):
                    # Write summary to file
                    self.writer.add_summary(summary, self.iteration)
                # Increase iteration counter
                self.iteration += 1
            # Display
            if not silence:
                display_iterations += 1
                if display_iterations % display_interval == 0:
                    print("*", end = "")
        if not silence:
            print(" Done")

    def change_epsilon(self):
        """Changes the epsilon for e-greedy exploration as a function of episode number"""
        self.agent.epsilon = self.epsilon_func(self.iteration)

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

    def _chunk(self, l, n):
        """
        Yield successive n-sized chunks from l
        Taken from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        """
        for i in range(0, len(l), n):
            yield l[i:i + n]