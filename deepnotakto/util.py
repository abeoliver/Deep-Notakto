#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: util.py                                                      #
#  Abraham Oliver, 2018                                               #
#######################################################################

import numpy as np
import pickle, hashlib
from time import time, localtime, asctime
from numpy.random import binomial, random

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
    n = np.zeros(x.shape, x.dtype)
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

def record(filename, agent, quality = None):
    n = agent.name
    a = str(agent.layers)
    g = str(agent.gamma)
    if agent.beta != None:
        b = str(agent.beta)
    else:
        b = "N\A"
    lr = str(agent.trainer.params["learn_rate"])
    i = str(agent.trainer.iteration)
    if quality != None:
        q = str(quality)
    else:
        q = "N/A"
    s = "{name}{nsp}{arch}{asp}{gamma}{gsp}{beta}{bsp}{learn}{lrsp}{iteration}{isp}{wins}\n".format(
        name = n, arch = a, gamma = g, beta = b, learn = lr, iteration = i, wins = q, nsp = " "* (26 - len(n)),
        asp = " " * (40 - len(a)), gsp = " " * (10 - len(g)), bsp = " " * (9 - len(b)), lrsp = " " * (10 - len(lr)),
        isp = " " * (13 - len(i))
    )
    with open(filename, 'a') as f:
        f.write(s)

def update(filename, agent, quality = None):
    # Remove the last line
    remove_last_record(filename)
    # Add a new record
    record(filename, agent, quality)

def remove_last_record(filename):
    # Read lines
    with open(filename, "r") as f:
        lines = f.readlines()
    # Rewrite
    with open(filename, "w") as f:
        for line in lines[:-1]:
            f.write(line)

def new_record_file(name):
    with open(name, 'w') as f:
        f.writelines(["Experimental Trials : {name}\n".format(name = name),
                      "The Deep Notakto Poject\n",
                      "NAME                      ARCHITECTURE                            GAMMA     BETA     L-RATE    ITERS        QUALITY\n",
                      ("=" * 120) + "\n"])

def unitize(x):
    """Normalize an array"""
    xmax, xmin = x.max(), x.min()
    # Catch divide by zero
    if xmax == xmin:
        return x
    return (x - xmin) / (xmax - xmin)

def seconds_to_time(seconds):
	minutes = seconds // 60
	return [minutes // 60, minutes % 60, seconds % 60]

def elapsed_time(start):
	new_time = time() - start
	elapsed = [int(i) for i in seconds_to_time(new_time)]
	clock = localtime(time())[3:5]
	return "Elapsed {} : {} : {} (at {} : {} {})".format(
        elapsed[0], str(elapsed[1]).zfill(2), str(elapsed[2]).zfill(2),
        int(clock[0]) % 12 if int(clock[0]) not in [0, 12] else 12,
        str(clock[1]).zfill(2), "PM" if clock[0] >= 12 and int(clock[0]) != 0 else "AM")

def unique_classifier():
    return hashlib.sha1(str(asctime()).encode("utf-8")).hexdigest()[:5]

def load_agent(filename, CLASS):
    with open(filename, "rb") as f:
        loaded = pickle.load(f)
        return CLASS(**loaded)

def array_in_list(ar, l):
    for i in l:
        if np.array_equal(ar, i):
            return True
    return False

def softmax(x):
    exped = np.exp(x - np.max(x))
    soft = exped / exped.sum()
    # To ensure sum to zero, offset each value by amount needed to sum to zero
    actual_sum = np.sum(soft)
    if actual_sum == 1.0:
        return soft
    return soft + (np.ones(soft.shape) * ((1.0 - actual_sum) / soft.size))

def create_board(index, b_size = 3):
    x = np.zeros([b_size, b_size], dtype = np.int8)
    if type(index) != list:
        x[index // b_size, index % b_size] = 1
    else:
        for i in index:
            x[i // b_size, i % b_size] = 1
    return x

def _clean(x, thresh):
    x[np.abs(x) < thresh] = 0.0
    return x

def move_to_vec(move, size):
    """ Translate a index-based move to a matrix-based move"""
    x = np.zeros([size * size], dtype = np.int32)
    x[move] = 1
    return np.reshape(x, [size, size])

def rotate_move(move, size, cw = False):
    """ Rotate an index-based move (if cw, rotate clockwise, otherwise counter cw) """
    if cw: return int(size * (1 + move % size) - 1 - move // size)
    return int(size * (size - 1 - (move % size)) + (move // size))

def reflect_move(move, size):
    """ Reflects an index-based move across the diagonal """
    return int((move % size) * size + (move // size))

def isomorphic_matrix(m1, m2):
    """ Checks if two matricies are isomorphic by rotation and reflection """
    # Check all isomorphisms
    for _ in range(4):
        if np.array_equal(m1, m2) or np.array_equal(m1.T, m2):
            return True
        # Rotate the target (rotates back to identity before it moves on)
        m1 = rotate(m1)
    return False

def average_value(agent, n = 100, p = 0):
    s = 0
    for _ in range(n):
        if p == 0:
            s += agent.value(binomial(1, random(), size = agent.shape))
        else:
            s += agent.value(binomial(1, p, size = agent.shape))
    return s / n