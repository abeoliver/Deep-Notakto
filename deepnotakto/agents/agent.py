# agent.py
# Abraham Oliver, 2017
# Deep-Notakto Project
from numpy.random import normal

class Agent (object):
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.episode = []
        self.episode_lengths = []
    
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

    def train(self, **kwargs):
        pass

    def record(self, state, action, reward):
        """Record an action"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def new_episode(self):
        """Reset the memory of an agent for a new episode"""
        self.states = []
        self.actions = []
        self.rewards = []

    def add_episode(self, state, action, reward):
        """Adds a move of a game to a game episode"""
        self.episode.append((state, action, reward))

    def save_episode(self, use_final = True, reward = None):
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
        # Fetch final reward
        final_reward = self.episode[-1][2]
        # Loop through each item in episode
        for s, a, r in self.episode:
            # Add to game record
            self.states.append(s)
            self.actions.append(a)
            # If using final reward, record the final reward
            if use_final:
                self.rewards.append(final_reward)
            # If using given reward, record that reward
            elif reward != None:
                self.rewards.append(reward)
            # Use reward in episode otherwise
            else:
                self.rewards.append(r)
        # Add the length of the episode to find episode later
        self.episode_lengths.append(len(self.episode))
        # Clear episode
        self.episode = []

    def reset_episode(self):
        """Reset the current episode"""
        self.episode = []

    def get_last_episode(self):
        """
        Get most recent episode
        Returns:
            List of (state, action, reward) tuples
        """
        b = self.episode_lengths[-1]
        return zip(self.states[-b:], self.actions[-b:], self.rewards[-b:])

    def get_i_episode(self, i):
        """
        Get the ith episode of the game records
        Parameters:
            i (int) - Index of episode to fetch
        Returns:
            List of (state, action, reward) tuples
        """
        a = sum(self.episode_lengths[:i])
        b = sum(self.episode_lengths[:i + 1])
        return zip(self.states[a:b], self.actions[a:b], self.rewards[a:b])

    def use_final_reward(self, epsiode_index = -1):
        """
        Use the final reward of an episode for the episode of the entire episode
        Parameters:
            epsiode_index (int) - Epsiode to change (default most recent)
        """
        # Get indexes of desired episode
        length = self.episode_lengths[epsiode_index]
        # Get starting index
        start = sum(self.episode_lengths[:i])
        # Get final reward
        reward = self.rewards[start + length - 1]
        # Apply to all in episode
        self.rewards[start : start + le - 1] = [reward] * (start + le - 1)

    def save(self, **kwargs):
        """Saves an agent to a file"""
        pass