#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: statistics.py                                                #
#  Abraham Oliver, 2018                                               #
#######################################################################

from pickle import load

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


def get_stats(name, add_suffix = False):
    """ Get a statistics file """
    with open(name + ".stats" if add_suffix else name, "rb") as f:
        s = load(f)
    return s


def group_stats_by_type(stats):
    """ Group statistics by the stat rather than by the iteration """
    # Get the list of stat types from an element
    if list(stats.keys())[0] == "meta":
        meta = True
        types = stats[list(stats.keys())[1]].keys()
        keys = sorted(list(stats.keys())[1:])
    else:
        meta = False
        types = stats[list(stats.keys())[0]].keys()
        keys = sorted(list(stats.keys()))
    # Prepare final dictionary to return
    final = {i: [] for i in types}
    final["iteration"] = []
    # Loop through all elements
    for key in keys:
        # Add the iteration as a stat
        final["iteration"].append(key)
        # Pull all data and put in appropriate group
        for t in types:
            final[t].append(stats[key][t])
    # Add in metadata
    if meta:
        final["meta"] = stats["meta"]
    return final


def plot_value(stats, values, fig_size = (10, 4), best_fit = True, **kwargs):
    """ Plot a given value from the satistics dictionary """
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
