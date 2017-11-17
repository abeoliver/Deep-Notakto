# main.py
# Abraham Oliver, 2017
# Deep Notakto Project

import os
from environment import Env
from agents.Q import Q
from agents.random_agent import RandomAgent
import util
from train import train_agent, ef

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

e = Env(3)
train_dict = {"mode": "episodic", "learn_rate": 1e-8, "rotate": True,
              "epsilon_func": lambda x: 0, "epochs": 1, "batch_size": -1}
a = Q([9, 100, 9], gamma = .7, epsilon = 1.0, training = train_dict,
      name = "debug", keras = False)
#a = util.load_agent("agents/saves/Q(22a61).npz", Q)
#a.training_params(training = train_dict)
#a.trainer.iteration = 0
r = RandomAgent(e)
a.mop = False
train_agent(e, a, r, -1, 100, save_a1 = False, save_a2 = False,
            path = "agents/saves/",
            record_name = "BB2.txt")