#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: data_generation.py                                           #
#  Abraham Oliver, 2018                                               #
#######################################################################

import os
from deepnotakto.train import train_model_with_tournament_evaluation
from deepnotakto.util import load_agent
from deepnotakto.notakto import QTree, Env, RandomAgent, measure
from random import choice, shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Layer architectures
# hiddens = [[10, 10, 10], [100], [100, 100], [1000], [1000, 1000],
#           [100, 100, 100], [], [10], [500, 500], [100, 100, 100]]
# SELF-PLAY VARS
# Number of simulations to run for each move
# simulations = [100, 100, 200, 500, 500, 1000, 1000, 1500, 2000, 2000]

hiddens = [[100, 100]]
simulations = [1000]

if __name__ == "__main__":
    path = "death/"
    tb_interval = 0
    tb_path = "death/"
    # TRAINING VARS
    queue_size = 300
    learn_rate = .001
    batch_size = 10
    replay_size = 100
    epochs = 10
    game_size = 3
    # Shuffle hyperparameters
    shuffle(hiddens)
    shuffle(simulations)
    for version in range(len(hiddens)):
        # Choose hyperparameters for this run
        hidden = hiddens[version]
        sims = simulations[version]

        print("Trial #{}\nSimulations - {}\nHidden - {}\n".format(
            version + 1, sims, hidden))

        # Add file extensions
        name = "{}".format(version + 1)
        model_path = path + name + ".npz"
        best_model_path = path + name + "_best.npz"
        stats_path = path + "{}.stats".format(name)
        params = {"rotate_live": True, "learn_rate": learn_rate,
                  "batch_size": batch_size, "replay_size": replay_size,
                  "epochs": epochs}
        agent = QTree(game_size, hidden, guided = True, player_as_input = True,
                      params = params, max_queue = queue_size, name = name,
                      tensorboard_interval = tb_interval,
                      tensorboard_path = tb_path, activation_func = "relu")

        statistics = {"meta": {"games": 100, "simulations": sims}}

        # Run the training loop
        env = Env(game_size)
        train_model_with_tournament_evaluation(
            agent = agent, env = env, opponent = RandomAgent(env),
            model_path = model_path, stats_path = stats_path,
            best_model_path = best_model_path, statistics = statistics,
            player = 1, sims = sims, console = True, iter_limit = 200,
            measure_func = measure)