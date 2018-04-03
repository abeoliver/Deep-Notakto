#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: server.py                                                    #
#  Abraham Oliver, 2018                                               #
#######################################################################

import os, sys
from pickle import load
sys.path.insert(0, '..')
from train import train_model_with_tournament_evaluation
from util import load_agent
from notakto import QTree, Env, measure, RandomAgent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# MODEL VARS
# Should load model or not
load_model = False
# Iteration of trials
version = 100
# Agent name
name = "{}".format(version)
path = "/voltorb/abraoliv/4x4/"
# Activation function
activation_func = "relu"
activation_type = "hidden"
# Layer architecture
game_size = 4
hidden = [1000, 1000, 200]
# Desired player evaluation
player = 2
# Guided or regular
guided = False

# TRAINING VARS
queue_size = 200
learn_rate = .005
batch_size = 30
replay_size = 100
epochs = 10

# Tensorboard checkpoint path and interval
tb_interval = 1
tb_path = "/voltorb/abraoliv/tensorboard/"

# SELF-PLAY VARS
# Number of simulations to run for each move
sims = 10000
# Number of self-play games to run
save_every = 1

# EVALUATION VARS
# Number of tournament games to run
games = 1000

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
        # Create a new agent with the given training/architecture parameters
        params = {"rotate_live": True, "learn_rate": learn_rate,
                  "batch_size": batch_size, "replay_size": replay_size,
                  "epochs": epochs}
        agent = QTree(game_size, hidden, guided = guided,
                      player_as_input = True, params = params,
                      max_queue = queue_size, name = name,
                      tensorboard_interval = tb_interval,
                      tensorboard_path = tb_path,
                      activation_func = activation_func,
                      activation_type = activation_type)
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
    env = Env(game_size)
    train_model_with_tournament_evaluation(agent = agent,
                                           opponent = RandomAgent(env),
                                           env = env,
                                           statistics = statistics,
                                           model_path = model_path,
                                           stats_path = stats_path,
                                           best_model_path = best_model_path,
                                           save_every = save_every,
                                           player = player,
                                           games = games,
                                           sims = sims,
                                           measure_func = measure)
