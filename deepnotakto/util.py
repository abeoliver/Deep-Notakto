#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: util.py                                                      #
#  Abraham Oliver, 2018                                               #
#######################################################################

import numpy as np
import pickle, hashlib
from time import time, localtime, asctime
from numpy.random import binomial, random


def load_agent(filename, CLASS):
    """ Load an agent of a given class from a file with a duplicative dict """
    with open(filename, "rb") as f:
        loaded = pickle.load(f)
        return CLASS(**loaded)


def rotate(x):
    """ Rotates a square array counter-clockwise """
    n = np.zeros(x.shape, x.dtype)
    for i in range(x.shape[0]):
        n[:, i] = x[i][::-1]
    return n


def rotate_batch(states, actions, rewards):
    """ Rotates set of states and actions """
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


def array_in_list(ar, l):
    """ Is a given numpy array in a list of arrays """
    for i in l:
        if np.array_equal(ar, i):
            return True
    return False


def unitize(x):
    """ Normalize an array """
    xmax, xmin = x.max(), x.min()
    # Catch divide by zero
    if xmax == xmin:
        return x
    return (x - xmin) / (xmax - xmin)


def seconds_to_time(seconds):
    """ Convert a count of seconds to hours, mins, secs """
    minutes = seconds // 60
    return [minutes // 60, minutes % 60, seconds % 60]


def elapsed_time(start):
    """ Pretty print the elapsed time since a time for timing output """
    new_time = time() - start
    elapsed = [int(i) for i in seconds_to_time(new_time)]
    clock = localtime(time())[3:5]
    return "Elapsed {} : {} : {} (at {} : {} {})".format(
        elapsed[0], str(elapsed[1]).zfill(2), str(elapsed[2]).zfill(2),
        int(clock[0]) % 12 if int(clock[0]) not in [0, 12] else 12,
        str(clock[1]).zfill(2), "PM" if clock[0] >= 12 and int(clock[0]) != 0 else "AM")


def unique_classifier():
    """ Get a (semi)unique classifier for an object"""
    return hashlib.sha1(str(asctime()).encode("utf-8")).hexdigest()[:5]


def softmax(x):
    exped = np.exp(x - np.max(x))
    soft = exped / exped.sum()
    # To ensure sum to zero, offset each value by amount needed to sum to zero
    actual_sum = np.sum(soft)
    if actual_sum == 1.0:
        return soft
    return soft + (np.ones(soft.shape) * ((1.0 - actual_sum) / soft.size))


def create_board(index, b_size = 3):
    """ Design a notakto board with given placed pieces """
    x = np.zeros([b_size, b_size], dtype = np.int8)
    if not isinstance(index, list):
        x[index // b_size, index % b_size] = 1
    else:
        for i in index:
            x[i // b_size, i % b_size] = 1
    return x


def move_to_vec(move, size):
    """ Translate a index-based move to a matrix-based move """
    x = np.zeros([size * size], dtype = np.int32)
    x[move] = 1
    return np.reshape(x, [size, size])


def chunk(l, n):
    """
    Yield successive n-sized chunks from l
    Taken from https://stackoverflow.com/questions/312443/
                how-do-you-split-a-list-into-evenly-sized-chunks
    """
    if n < 0:
        yield l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def rotate_move(move, size, cw = False):
    """ Rotate an index-based move (if cw, rotate clockwise, otherwise ccw) """
    if cw:
        return int(size * (1 + move % size) - 1 - move // size)
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


def bin_to_array(b):
    """ Convert a binary state representation to an array """
    a = np.array([int(i) for i in str(b)])
    size = int(np.sqrt(a.size))
    return np.reshape(a, [size, size])


def array_to_bin(a):
    """ Convert an array state representation to binary """
    return ''.join([str(int(i)) for i in np.reshape(a, -1).tolist()])


def get_move_dict(name, size):
    """ Get a move dictionary from a file """
    with open(name) as f:
        d = {}
        for line in f:
            parts = [str(bin(int(i)))[2:].zfill(size * size)
                     for i in line.rstrip().split()]
            d[parts[0]] = bin_to_array(parts[1])
    return d
