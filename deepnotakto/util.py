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

def bin_to_array(b):
    a = np.array([int(i) for i in str(b)])
    size = int(np.sqrt(a.size))
    return np.reshape(a, [size, size])

def array_to_bin(a):
    return ''.join([str(int(i)) for i in np.reshape(a, -1).tolist()])

def get_move_dict(name, size):
    with open(name, "r") as f:
        d = {}
        for line in f:
            parts = [str(bin(int(i)))[2:].zfill(size * size) for i in line.rstrip().split()]
            d[parts[0]] = bin_to_array(parts[1])
    return d