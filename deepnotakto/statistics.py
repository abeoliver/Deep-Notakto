# statistics.py
# Abraham Oliver, 2018
# Deep-Notakto

import matplotlib.pyplot as plt
from pickle import load
import numpy as np
from sklearn import linear_model

def get_stats(name, add_suffix = True):
    with open(name + ".stats" if add_suffix else name, "rb") as f:
        s = load(f)
    return s

def group_stats_by_type(stats):
    # Get the list of stat types from an element
    types = stats[list(stats.keys())[0]].keys()
    # Prepare final dictionary to return
    final = {i: [] for i in types}
    final["iteration"] = []
    # Loop through all elements
    for key in sorted(list(stats.keys())):
        # Add the iteration as a stat
        final["iteration"].append(key)
        # Pull all data and put in appropriate group
        for t in types:
            final[t].append(stats[key][t])
    return final

def plot_value(stats, values, fig_size = (10, 4), best_fit = True, **kwargs):
    if type(values) == str:
        values = [values]
    plt.figure(figsize = fig_size)
    for i, val in enumerate(values):
        plt.subplot((len(values) // 2) + len(values) % 2,
                    2 if len(values) > 1 else 1,
                    i + 1)
        best_fit_args = []
        if best_fit:
            model = linear_model.LinearRegression()
            model.fit(np.array(stats["iteration"]).reshape(-1, 1),
                      np.array(stats[val]).reshape(-1, 1))
            best_fit_args.append([[0], [max(stats["iteration"])]])
            best_fit_args.append(model.predict([[0], [max(stats["iteration"])]]))
        plt.plot(stats["iteration"], stats[val], *best_fit_args, **kwargs)
        plt.ylabel(val.title().replace("_", " "))
    plt.show()
    return None

def measure(agent, **kwargs):
    # Use the given stats as a starting point
    stats = kwargs
    # Zero board
    z = np.zeros(agent.shape)
    # Value of zero board
    stats["zero_val"] = agent.value(z)
    # Size of raw predictions of zero board
    stats["zero_norm"] = np.linalg.norm(agent.raw(z))
    # Maximum probability on zero board
    policy = agent.policy(z)
    stats["zero_max"] = np.max(policy)
    # Mean of probabilities on zero board
    stats["zero_mean"] = np.mean(policy)
    return stats

def measure_generator_learner(generator, learner, **kwargs):
    # Use the given stats as a starting point
    stats = kwargs
    # Zero board
    z = np.zeros(generator.shape)
    # Value of zero board
    stats["gen_zero_val"] = generator.value(z)
    stats["learn_zero_val"] = learner.value(z)
    # Size of raw predictions of zero board
    stats["gen_zero_norm"] = np.linalg.norm(generator.raw(z))
    stats["learn_zero_norm"] = np.linalg.norm(learner.raw(z))
    # Maximum probability on zero board
    gen_policy = generator.policy(z)
    learn_policy = learner.policy(z)
    stats["gen_zero_max"] = np.max(gen_policy)
    stats["learn_zero_max"] = np.max(learn_policy)
    # Mean of probabilities on zero board
    stats["gen_zero_mean"] = np.mean(gen_policy)
    stats["learn_zero_mean"] = np.mean(learn_policy)
    return stats