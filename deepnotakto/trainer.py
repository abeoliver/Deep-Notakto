# trainer.py
# Abraham Oliver, 2017
# Deep-Notakto Project

import numpy as np
import tensorflow as tf
from random import shuffle

class Trainer (object):
    def __init__(self, agent):
        """
        Initializes a Trainer object
        Parameter:
            agent (Q.Q) - Q Agent to train
        """
        self.agent = agent
        self.iteration = 0
        self.writer = tf.summary.FileWriter("tensorboard/" + agent.name,
                                            agent.session.graph)

    def get_online(self, learn_rate = .0001):
        """Gets a callable function of online with a given learning rate"""
        return lambda x, y, z: self.online(x, y, z, learn_rate)

    def online(self, state, action, reward, learn_rate = .0001):
        """
        Train the on a single state and reward (usually in an environment)
        Parameters:
            state ((N, N) array) - State
            action ((N, N) array) - Action applied
            reward (float) - Reward for action on state
            learn_rate (float) - Learning rate
        """
        target = self.agent.target(state, action, self.agent.get_Q(state), reward)
        self.agent.update([state], [target], learn_rate)

    def offline(self, states, actions, rewards, batch_size = 1, epochs = 1,
                learn_rate = .0001, silence = False):
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
        Note:
            Targets are caculated at the beginning of each epoch.
            Therefore, all targets in a given epochs use the same
            Q-function and are not effected by the others in the
            batch
        """
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
            # Batch train once
            self.batch(states, targets, batch_size, 1, learn_rate, True)
            # Output
            if not silence:
                if epoch % display_interval == 0:
                    print("*", end = "")
        if not silence:
            print(" Done")

    def batch(self, states, targets, batch_size, epochs = 1,
              learn_rate = .0001, silence = False):
        """
        Trains the agent over a batch of states and targets
        Parameters:
            states (List of (N, N) arrays) - List of states
            targets (List of (N, N) arrays) - List of targets for each state
            batch_size (int) - Number of samples in each minibatch
            epochs (int) - Number of iterations over the entire dataset
            learn_rate (float) - Learning rate
            silence (bool) - Print to stdout?
        """
        # Output
        if not silence:
            print("Training ", end="")
            eb = (epochs * batch_size)
            display_interval = eb// 10 if eb >= 10 else 1
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
                # Increase iteration counter
                self.iteration += 1
                # Get the states and targets from the indidicies of the
                # batch and pass into update
                summary = self.agent.update([states[b] for b in batch],
                                             [targets[b] for b in batch],
                                             learn_rate)
                # Write summary to file
                self.writer.add_summary(summary, self.iteration)
                # Display
                if not silence:
                    display_iterations += 1
                    if display_iterations % display_interval == 0:
                        print("*", end = "")
        if not silence:
            print(" Done")

    def change_epsilon(self, func = lambda x: 1.0 / (x + 1)):
        """Changes the epsilon for e-greedy exploration as a function of episode number"""
        self.agent.epsilon = func(self.iteration)

    def get_rotations(self, states, targets):
        """Train over the rotated versions of each state and reward"""
        # Copy states so as not to edit outside of scope
        states = copy(states)
        # Reshape targets for rotation
        targets = [np.reshape(t, states[0].shape) for t in targets]
        # Collect rotated versions of each state and target
        new_tates = []
        new_targets = []
        print("Rotating ... ", end = "")
        for s, t in zip(states, targets):
            ns = s
            nt = t
            for i in range(3):
                rs = self.rotate(ns)
                rt = self.rotate(nt)
                new_states.append(rs)
                new_targets.append(np.reshape(rt, -1))
                ns = rs
                nt = rt
        print("Done")
        # Combine lists
        all_states = states + new_states
        all_targets = targets + new_targets
        return [allStates, allTargets]

    def _chunk(self, l, n):
        """
        Yield successive n-sized chunks from l
        Taken from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        """
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def rotate(self, x):
        """Rotates an array counter-clockwise"""
        n = np.zeros(x.shape)
        for i in range(x.shape[0]):
            n[:, i] = x[i][::-1]
        return n