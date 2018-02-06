# server.py
# Abraham Oliver, 2017
# Deep Notakto Project

# THING TO TRY: GENERATE REPLAYS THEN TRAIN INSTEAD OF LIVE TRAININING

import os
import sys

from time import time, localtime
from pickle import dump, load
from numpy import sqrt

sys.path.insert(0, '..')
from util import load_agent, average_value, seconds_to_time
from train import tournament
from agents.random_agent import RandomAgent
from environment import Env
from agents.qtree import QTree
from statistics import measure

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MODEL VARS
# Should load model or not
load_model = False
# Agent name
name = "Goal4x4_5"
# Path to save agent to
model_path = "/voltorb/abraoliv/goal_4x4_5"
# Activation function
activation_func = "swish"
activation_type = "hidden"
# Layer architecture
architecture = [16, 500, 500, 500, 500, 16]
# Desired player evaluation
player = 2

# TRAINING VARS
queue_size = 100
learn_rate = .00005
batch_size = 5
replay_size = 40
epochs = 10

# Tensorboard checkpoint path and interval
tb_interval = 10
tb_path = "/voltorb/abraoliv/tensorboard/"

# SELF-PLAY VARS
# Number of simulations to run for each move
sims = 200
# Number of self-play games to run
save_every = 20

# EVALUATION VARS
# Should evaluate by tournament
evaluate = True
# Number of tournament games to run
games = 100
# Evaluate against best
eval_against_best = True
# Should save tournament statistics
save_stats = True
# File path for statistics
stats_path = "/voltorb/abraoliv/{}".format(name)
# Should log console output or not
log = False
# File path for log
log_path = "/voltorb/abraoliv/training"

def elapsed_time(start):
	new_time = time() - start
	elapsed = [int(i) for i in seconds_to_time(new_time)]
	clock = localtime(time())[3:5]
	return "Time                  {} : {} : {} (at {} : {} {})".format(
        elapsed[0], str(elapsed[1]).zfill(2), str(elapsed[2]).zfill(2),
        int(clock[0]) % 12 if int(clock[0]) not in [0, 12] else 12, str(clock[1]).zfill(2),
        "PM" if clock[0] >= 12 and int(clock[0]) != 0 else "AM")

if __name__ == "__main__":
    # Environment to play in
    env = Env(int(sqrt(architecture[0])))
    # Opponent to play against
    opponent = RandomAgent(env)

    # Get path for best model save
    best_model_path = model_path + "_best.npz"
    # Value for comparing models
    prev_best_model_val = 0
    # Current best model
    current_best = opponent

    # Add file extensions
    model_path += ".npz"
    stats_path += ".stats"
    log_path += ".log"

    # Create or load model
    if load_model:
        print("Loading model...", end = " ")
        # Load an agent from the model path and with the given class definition
        agent = load_agent(model_path, QTree)
    else:
        print("Building model...", end = " ")
        # Create a new randomly-initialized agent with the given training/architecture parameters
        params = {"rotate_live": True, "learn_rate": learn_rate, "batch_size": batch_size,
                  "replay_size": replay_size, "epochs": epochs}
        agent = QTree(architecture, params = params, max_queue = queue_size, name = name,
                      tensorboard_interval = tb_interval, tensorboard_path = tb_path,
                      activation_func = activation_func, activation_type = activation_type)
    print("Complete")

    # Create or load statistic set
    if load_model:
        # Get the current statistic set
        with open(stats_path, "rb") as f:
            statistics = load(f)
    else:
        # Create a new statistic set
        statistics = {}

    # Begin console output with save location and time
    clock = localtime(time())[3:5]
    o = "\n\n-------- {} --------\nSaved as '{}'\nStarted at {} : {} {}\n".format(
        agent.name, model_path, int(clock[0]) % 12 if clock[0] not in [0, 12] else 12,
        str(clock[1]).zfill(2), "PM" if clock[0] >= 12 and int(clock[0]) != 0 else "AM")
    print(o)
    # Write preamble to file
    if log:
        with open(log_path, "a") as f: f.write(o)

    # Start clock
    start = time()

    # Main training loop
    while True:
        # Run self play training algorithm
        print("Self play... ", end = "")
        agent.self_play(games = save_every, simulations = sims)
        print("Completed")
        # Save the model
        agent.save(model_path)
        # Prepare log/console output
        outputs = []
        outputs.append("TIME PLACEHOLDER")
        outputs.append("Iteration             {}".format(agent.iteration))
        # Evaluation tournament
        if evaluate:
            # Q-based evaluation tournament
            agent.mode("q")
            print("Q-based evaluation... ", end = "")
            if player == 1:
                q_wins = tournament(env, agent, opponent, games)[0]
            else:
                q_wins = tournament(env, opponent, agent, games)[1]
            print("Complete")
            outputs.append("Q Evaluation          {}%".format(int(q_wins * 100 / games)))

            # Best Tournament
            best_wins_1 = 0
            best_wins_2 = 0
            if eval_against_best:
                print("Against current best evaluation... ", end = "")
                best_wins_1 = tournament(env, agent, current_best, games // 2)[0]
                best_wins_2 = tournament(env, current_best, agent, games // 2)[1]
                print("Complete")

            # Designate as best model if is better than previous model
            is_best = False
            if ((q_wins / games) + .3 * ((best_wins_1 + best_wins_2) / games)) > prev_best_model_val:
                prev_best_model_val = ((q_wins / games) + .3 * ((best_wins_1 + best_wins_2) / games))
                agent.save(best_model_path)
                is_best = True
                current_best = agent
                outputs.append("BEST MODEL")

            # Save statistics
            statistics[agent.iteration] = measure(agent, q_wins = q_wins, time = time() - start,
                                                  best = is_best, best_wins_1 = best_wins_1,
                                                  best_wins_2 = best_wins_2, best_model_val = prev_best_model_val)
        else:
            # Save statistics
            statistics[agent.iteration] = measure(agent, time = time() - start)

        # Fill in placeholders
        outputs[0] = elapsed_time(start)
        # Construct final output string
        output = "".join(i + "\n" for i in outputs)
        # Output to console
        print(output)
        # Output to file
        if log:
            with open(log_path, "a") as f:
                f.write("\n" + output)
        # Save statistics
        with open(stats_path, "wb") as f:
            dump(statistics, f)