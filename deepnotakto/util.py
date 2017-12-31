# util.py
# Abraham Oliver, 2017
# Deep Notakto Project

import numpy as np
import pickle, hashlib
from time import time, localtime, asctime

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
    a = str(agent.architecture)
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

def normalize(x):
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
	return "Elapsed {} : {} : {} (at {} : {})".format(elapsed[0], elapsed[1],
													 elapsed[2], clock[0],
													 clock[1])

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