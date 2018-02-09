# server.py
# Abraham Oliver, 2017
# Deep Notakto Project

# THING TO TRY: GENERATE REPLAYS THEN TRAIN INSTEAD OF LIVE TRAININING

import os
from deepnotakto.train import train_model_with_tournament_evaluation
from deepnotakto.util import load_agent
from deepnotakto.agents.qtree import QTree

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MODEL VARS
# Should load model or not
load_model = False
# Iteration of trials
version = 8
# Agent name
name = "Goal4x4_{}".format(version)
model_path = "/voltorb/abraoliv/goal_4x4_{}".format(version)
# Activation function
activation_func = "relu"
activation_type = "hidden"
# Layer architecture
game_size = 4
hidden = [700, 500, 400, 300, 200]
# Use player as an input
player_as_input = True
# Desired player evaluation
player = 2

# TRAINING VARS
queue_size = 100
learn_rate = .001
batch_size = 10
replay_size = 40
epochs = 10

# Tensorboard checkpoint path and interval
tb_interval = 10
tb_path = "/voltorb/abraoliv/tensorboard/"

# SELF-PLAY VARS
# Number of simulations to run for each move
sims = 150
# Number of self-play games to run
save_every = 20

# EVALUATION VARS
# Number of tournament games to run
games = 100
# File path for statistics
stats_path = "/voltorb/abraoliv/{}".format(name)

if __name__ == "__main__":
    # Add file extensions
    best_model_path = model_path + "_best.npz"
    model_path += ".npz"
    stats_path += ".stats"

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
        agent = QTree(game_size, hidden, player_as_input = player_as_input,
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
        statistics = {}

    # Run the training loop
    train_model_with_tournament_evaluation(agent = agent,
                                           model_path = model_path,
                                           stats_path = stats_path,
                                           best_model_path = best_model_path,
                                           statistics = statistics,
                                           save_every = save_every,
                                           player = player,
                                           games = games)