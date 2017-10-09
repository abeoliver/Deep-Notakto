# util.py
# Abraham Oliver, 2017
# Deep Notakto Project

import numpy as np
from agents.Q import Q
import pickle

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

def load_q_agent(filename):
    with open(filename, "rb") as f:
        loaded = pickle.load(f)
        q = Q(loaded["params"]["layers"], gamma = loaded["params"]["gamma"],
              name = loaded["params"]["name"], initialize = False)
        q.init_model(w = loaded["weights"], b=loaded["biases"])
        q._init_training_vars()
        return q

def record(file, agent, trainer, quality = None):
    a = str(agent.layers)
    g = str(agent.gamma)
    b = str(agent.beta)
    lr = str(trainer.learn_rate)
    i = str(trainer.iteration)
    q = str(quality)
    s = "\n{arch}{asp}{gamma}{gsp}{beta}{bsp}{learn}{lrsp}{iteration}{isp}{quality}".format(
        arch = a, gamma = g, beta = b, learn = lr, iteration = i, quality = q, asp = " " * (33 - len(a)),
        gsp = " " * (10 - len(g)), bsp = " " * (9 - len(b)), lrsp = " " * (17 - len(lr)), isp = " " * (14 - len(i)))
    with open(file, 'a') as f:
        f.write(s)

def new_record_file(name):
    with open(name, 'w') as f:
        f.writelines(["Experimental Trials : {name}\n".format(name = name),
                      "The Deep Notakto Poject\n",
                      "ARCHITECTURE                     GAMMA     BETA     LEARNING RATE    ITERATIONS    QUALITY\n",
                      ("=" * 118) + "\n"])

def norm(x):
    """Normalize an array"""
    xmax, xmin = x.max(), x.min()
    # Catch divide by zero
    if xmax == xmin:
        return x
    return (x - xmin) / (xmax - xmin)