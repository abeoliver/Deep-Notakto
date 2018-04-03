#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: train.py                                                     #
#  Abraham Oliver, 2018                                               #
#######################################################################

from time import time

import os

from time import time, localtime
from pickle import dump, load
from numpy import sqrt
from copy import deepcopy

from deepnotakto.util import load_agent, seconds_to_time


def play(env, a1, a2):
    """ Play two agents against eachother in an environment """
    a1.new_episode()
    a2.new_episode()
    # Reset environment
    env.reset()
    while True:
        # Get current player
        player = [a1, a2][env.turn % 2]
        # Play
        observation = player.act(env)
        # Check for illegal move or a win
        if observation["info"]["illegal"] or observation["winner"] != 0:
            break
    # End the game
    a1.save_episode()
    a2.save_episode()
    # Return winner
    return env.winner()


def tournament(env, a1, a2, games):
    """ Run a tournament environment for a given number of games """
    wins = [0, 0]
    for _ in range(games):
        wins[play(env, a1, a2) - 1] += 1
    return wins


def elapsed_time(start):
    """ Get a pretty print string for the time since a given start time """
    new_time = time() - start
    elapsed = [int(i) for i in seconds_to_time(new_time)]
    clock = localtime(time())[3:5]
    return "Time                  {} : {} : {} (at {} : {} {})".format(
        elapsed[0], str(elapsed[1]).zfill(2),
        str(elapsed[2]).zfill(2),
        int(clock[0]) % 12 if int(clock[0]) not in [0, 12] else 12,
        str(clock[1]).zfill(2),
        "PM" if clock[0] >= 12 and int(clock[0]) != 0 else "AM")


def train_model_with_tournament_evaluation(agent, opponent, env,
                                           statistics = {},
                                           model_path = "model.npz",
                                           stats_path = "model.stats",
                                           best_model_path = None,
                                           save_every = 1, sims = 100,
                                           player = 1, games = 100,
                                           break_at_100 = True, console = True,
                                           iter_limit = 0, measure_func = None):
    """
    Run a training loop with evaluation against an opponent

    Args:
        agent: (Agent) An agent to train
        opponent: (Agent) Agent to compete against in tournament evaluation
        env: (Env) Environment to compete and evaluate in
        statistics: (dict) Current statistics dictionary
        model_path: (string) Filepath for saving current model
        stats_path: (string) Filepath for saving statistic dictionary
        best_model_path: (string) Filepath for saving best model
        save_every: (int) Number of iterations between saves
        sims: (int) Number of simulations per move for treesearch
        player: (1 or 2) Player number for agent in tournament
        games: (int) Number of games in tournament
        break_at_100: (bool) Should stop training after 100% win rate
        console: (bool) Output progress to console?
        iter_limit: (int) Maximum iterations in training loop
        measure_func: (agent, kwargs -> dict) Function to calculate statistics
    """
    # Clean input
    if best_model_path is None:
        ext = model_path[-4:]
        best_model_path = model_path[:-4] + "_best" + ext
    if measure_func is None:
        measure_func = default_measure

    # Value for comparing best model and current model
    prev_best_model_val = -1

    # Begin console output with save location and time
    if console:
        clock = localtime(time())[3:5]
        o = "\n\n-------- {} --------\nSaved as '{}'\nStarted at {} : {} {}\n".\
            format(agent.name, model_path,
                   int(clock[0]) % 12 if clock[0] not in [0, 12] else 12,
                   str(clock[1]).zfill(2), "PM" if clock[0] >= 12 and
                                                   int(clock[0]) != 0 else "AM")
        print(o)

    # Start clock
    start = time()

    # Main training loop
    while True:
        # Run self play training algorithm
        if console:
            print("Self play... ", end = "")
        # Generate games
        agent.self_play(games = save_every, simulations = sims, train = False)
        # Train after generation
        agent.train()
        if console:
            print("Completed")
        # Save the model
        agent.save(model_path)
        # Prepare console output
        outputs = []
        outputs.append("TIME PLACEHOLDER")
        outputs.append("Iteration             {}".format(agent.iteration))

        # Reset environment
        env.reset()

        # Q-based evaluation tournament
        if console:
            print("Q-based evaluation... ", end = "")
        players = [agent, opponent]
        q_wins = tournament(env, players[player - 1],
                            players[2 - player], games)[player - 1]
        if console:
            print("Complete")
        outputs.append("Q Evaluation          {}%".format(
            int(q_wins * 100 / games)))

        # Designate as best model if is better than previous model
        is_best = False
        if (q_wins / games) > prev_best_model_val:
            prev_best_model_val = q_wins / games
            agent.save(best_model_path)
            is_best = True
            outputs.append("BEST MODEL")

        # Save statistics
        statistics[agent.iteration] = measure_func(
            agent, q_wins = q_wins, time = time() - start, best = is_best,
            best_model_val = prev_best_model_val)

        # Fill in placeholders
        outputs[0] = elapsed_time(start)
        # Construct final output string
        output = "".join(i + "\n" for i in outputs)
        if console:
            print(output)
        # Save statistics
        with open(stats_path, "wb") as f:
            dump(statistics, f)

        # End training if 100% achieved
        if break_at_100 and q_wins == games:
            return None

        # End training if iteration limit is passed
        if iter_limit > 0 and agent.iteration >= iter_limit:
            return None


def default_measure(agent, **given_stats):
    """ Default measure fuction; return given statistics in a dictionary """
    return given_stats
