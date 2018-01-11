# server.py
# Abraham Oliver, 2017
# Deep Notakto Project

import os, sys
sys.path.insert(0, '..')
from time import time, localtime
from numpy.random import binomial, random
from util import load_agent, average_value, seconds_to_time
from agents.qtree import QTree
from train import tournament
from agents.random_agent import RandomAgent
from environment import Env
from agents.treeactivated import TanhHidden

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Model Vars
load_model = True
name = "ServerTrained5x5"
model_path = "agents/saves/server_trained_5x5_jan_9.npz"
CLASS = TanhHidden
architecture = [25, 200, 200, 200, 25]
queue_size = 200

# Training Vars
learn_rate = .005
batch_size = 10
replay_size = 100
epochs =  2

# Self-Play Vars
sims = 20
save_every = 1

# Evaluation Vars
log_name = "training_plus_evaluation.log"
evaluate = True
play_simulations = 10
games = 10
env = Env(5)
opponent = RandomAgent(env)

def elapsed_time(start):
	new_time = time() - start
	elapsed = [int(i) for i in seconds_to_time(new_time)]
	clock = localtime(time())[3:5]
	return "Time                  {} : {} : {} (at {} : {} {})".format(
        elapsed[0], str(elapsed[1]).zfill(2), str(elapsed[2]).zfill(2),
        int(clock[0]) % 12 if int(clock[0]) not in [0, 12] else 12, str(clock[1]).zfill(2),
        "PM" if clock[0] >= 12 and int(clock[0]) != 0 else "AM")

if __name__ == "__main__":
    # Create or load model
    if load_model:
        print("Loading model...", end = " ")
        agent = load_agent(model_path, CLASS)
    else:
        print("Building model...", end = " ")
        params = {"rotate_live": True, "train_live": True, "learn_rate": learn_rate,
                  "batch_size": batch_size, "replay_size": replay_size, "epochs": epochs}
        agent = CLASS(architecture, params = params, play_simulations = play_simulations,
                      max_queue = queue_size, name = name)
    print("Complete")

    # Train and evaluate
    # Console and file output
    clock = localtime(time())[3:5]
    o = "\n\n-------- {} --------\nSaved as '{}'\nStarted at {} : {} {}\n".format(
        agent.name, model_path, int(clock[0]) % 12 if clock[0] not in [0, 12] else 12,
        str(clock[1]).zfill(2), "PM" if clock[0] >= 12 and int(clock[0]) != 0 else "AM")
    print(o)
    with open(log_name, "a") as f: f.write(o)
    # Start clock
    start = time()
    while True:
        print("Self play... ", end = "")
        agent.self_play(save_every, sims)
        print("Completed")
        agent.save(model_path)
        outputs = []
        outputs.append("TIME PLACEHOLDER")
        outputs.append("ITERATION PLACEHOLDER")
        outputs.append("AVG VALUE PLACEHOLDER")
        if evaluate:
            agent.mode("search")
            print("Search-based evaluation... ", end = "")
            sep1 = tournament(env, agent, opponent, games)[0]
            #sep2 = tournament(env, opponent, agent, games)[1]
            print("Complete")
            outputs.append("Search Evaluation     {}% ".format(int(sep1 * 100 / games)))
                                                                        #int(sep2 * 100 / games)))
            agent.mode("q")
            print("Q-based evaluation... ", end = "")
            qep1 = tournament(env, agent, opponent, games)[0]
            #qep2 = tournament(env, opponent, agent, games)[1]
            print("Complete")
            outputs.append("Q Evaluation          {}%".format(int(qep1 * 100 / games)))
                                                                        #int(qep2 * 100 / games)))
        # Fill in placeholders
        outputs[0] = elapsed_time(start)
        outputs[1] = "Iteration             {}".format(agent.iteration)
        outputs[2] = "Average Value         {:.3f}".format(average_value(agent, n=1000))
        # Construct final string
        output = "".join(i + "\n" for i in outputs)
        # Output to console
        print(output)
        # Output to file
        with open(log_name, "a") as f:
            f.write("\n" + output)