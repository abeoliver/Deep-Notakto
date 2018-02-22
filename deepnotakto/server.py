# server.py
# Abraham Oliver, 2017
# Deep Notakto Project

# THING TO TRY: GENERATE REPLAYS THEN TRAIN INSTEAD OF LIVE TRAININING

import os, sys
sys.path.insert(0, '..')
from train import train_model_with_tournament_evaluation,\
    train_generator_learner, train_model_only_best
from util import load_agent
from agents.qtree import QTree

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MODEL VARS
# Should load model or not
load_model = False
# Iteration of trials
version = 10
# Agent name
name = "{}".format(version)
path = "/voltorb/abraoliv/4x4/"
# Activation function
activation_func = "relu"
activation_type = "hidden"
# Layer architecture
game_size = 4
hidden = [1000, 500, 200]
# Desired player evaluation
player = 2

# TRAINING VARS
queue_size = 300
learn_rate = .001
batch_size = 10
replay_size = 100
epochs = 10

# Tensorboard checkpoint path and interval
tb_interval = 1
tb_path = "/voltorb/abraoliv/tensorboard/"

# SELF-PLAY VARS
# Number of simulations to run for each move
sims = 2500
# Number of self-play games to run
save_every = 1

# EVALUATION VARS
# Number of tournament games to run
games = 100

if __name__ == "__main__":
    # Add file extensions
    model_path = path + name + ".npz"
    best_model_path = path + name + "_best.npz"
    stats_path = path + "{}.stats".format(name)

    # Create or load model
    if load_model:
        print("Loading model...", end=" ")
        # Load an agent from the model path and with the given class definition
        agent = load_agent(model_path, QTree)
        agent.change_param("learn_rate", learn_rate)
    else:
        print("Building model...", end=" ")
        # Create a new randomly-initialized agent with the given training/architecture parameters
        params = {"rotate_live": True, "learn_rate": learn_rate, "batch_size": batch_size,
                  "replay_size": replay_size, "epochs": epochs}
        agent = QTree(game_size, hidden, player_as_input = True,
                      params = params, max_queue = queue_size, name = name,
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
        statistics = {"meta": {"games": games, "simulations": sims}}

    # Run the training loop
    train_model_with_tournament_evaluation(agent = agent,
                                           model_path = model_path,
                                           stats_path = stats_path,
                                           best_model_path = best_model_path,
                                           statistics = statistics,
                                           save_every = save_every,
                                           player = player,
                                           games = games,
                                           sims = sims)
    """train_generator_learner(agent = agent,
                            model_path = model_path,
                            stats_path = stats_path,
                            statistics = statistics,
                            challenge_every = save_every,
                            player = player,
                            games = games,
                            sims = sims)"""