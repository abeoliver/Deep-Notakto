# util.py
# Abraham Oliver, 2017
# Deep Notakto Project

import numpy as np

def save_set(name, states, actions, rewards):
    """Save a set of states, actions, and rewards"""
    with open(name, "wb") as f:
        np.savez(f, states = states, actions = actions, rewards = rewards)

def load_set(name):
    """Loads a set of states, actions, and rewards"""
    with open(name, "rb") as f:
        loaded = np.load(f)
        return [list(loaded["states"]),
                list(loaded["actions"]),
                list(loaded["rewards"])]

def rotate(x):
    """Rotates an array counter-clockwise"""
    n = np.zeros(x.shape)
    for i in range(x.shape[0]):
        n[:, i] = x[i][::-1]
    return n

def rotate_batch(states, actions, rewards):
    """Rotates all states and actions because game is rotation invariant"""
    new_states = list(states)
    new_actions = list(actions)
    new_rewards = list(rewards)
    for i in range(len(states)):
        s = states[i]
        a = actions[i]
        r = rewards[i]
        for i in range(3):
            ns = rotate(s)
            na = rotate(a)
            new_states.append(ns)
            new_actions.append(na)
            new_rewards.append(r)
            s = ns
            a = na
    return [new_states, new_actions, new_rewards]