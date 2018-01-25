# statistics.py
# Abraham Oliver, 2018
# Deep-Notakto

import matplotlib.pyplot as plt
from pickle import load
import numpy as np

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

def plot_value(stats, values, fig_size = (10, 4), **kwargs):
    if type(values) == str:
        values = [values]
    plt.figure(figsize = fig_size)
    for i, val in enumerate(values):
        plt.subplot((len(values) // 2) + len(values) % 2,
                    2 if len(values) > 1 else 1,
                    i + 1)
        plt.plot(stats["iteration"], stats[val], **kwargs)
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
    stats["zero_max"] = np.max(agent.policy(z))
    return stats