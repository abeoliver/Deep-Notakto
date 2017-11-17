# server.py
# Abraham Oliver, 2017
# Deep Notakto Project

import os
from environment import Env
from agents.Q import Q
from agents.random_agent import RandomAgent
import util
from train import train_agent, ef

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

e = Env(4)
train_dict = {"type": "episodic", "learn_rate": 1e-8, "rotate": True,
              "epsilon_func": lambda x: ef(x, 20000), "epochs": 1,
              "batch_size": -1}
a = Q([16, 100, 400, 16], gamma = .7, epsilon = 1.0,
       training = train_dict, tensorboard_interval = 0)
r = RandomAgent(e)
train_agent(e, r, a, -1, 1000, save_a1 = False, save_a2 = True,
            path = "agents/saves/")