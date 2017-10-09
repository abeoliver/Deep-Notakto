# train.py
# Abraham Oliver, 2017
# Deep Notakto Project

import os
from environment import Env
from trainer import Trainer
from agents.Q import Q
from agents.random_plus import RandomAgentPlus
from time import time, localtime
from random import choice

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_agent(env, episodes, p1, p2, t1 = None, t2 = None):
	start = time()

def elapsed_time(start):
	new_time = time() - start
	elapsed = [int(i) for i in seconds_to_time(new_time)]
	clock = localtime(time())[3:5]
	return "Elapsed {} : {} : {} (at {} : {})".format(elapsed[0], elapsed[1],
													 elapsed[2], clock[0],
													 clock[1])

def seconds_to_time(seconds):
	minutes = seconds // 60
	return [minutes // 60, minutes % 60, seconds % 60]



