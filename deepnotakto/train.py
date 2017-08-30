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

ITERATIONS = 1000000
SAVE_FREQ = 100
EVAL_FREQ = 10000

def train_agent(env, p1, p2, t1 = None, t2 = None):
	start = time()
	for i in range(ITERATIONS):
		if i % (ITERATIONS / 100) == 0:
			print("{}% [{} / {}] {}".format(
				int((i / ITERATIONS) * 100), i, ITERATIONS, elapsed_time(start)))
		env.play(p1, p2, 1, trainer_a1 = t1, trainer_a2 = t2,
				 final_reward = True, silence = True)
		if i % SAVE_FREQ == 0:
			if t1 != None:
				p1.save(name="/scratch/abraoliv/{}.npz".format(p1.name))
			if t2 != None:
				p2.save(name="/scratch/abraoliv/{}.npz".format(p2.name))
		if i % EVAL_FREQ == 0:
			evaluate(env, p1, p2, int(i / 10000), 1)
	print("{}% [{} / {}]  Elapsed : {}".format(100, ITERATIONS, ITERATIONS, elapsed_time(start)))
	if t1 != None:
		p1.save(name="/scratch/abraoliv/{}.npz".format(p1.name))
	if t2 != None:
		p2.save(name="/scratch/abraoliv/{}.npz".format(p2.name))
	print("Training Complete.")

def evaluate(env, agent, opponent, iters, player_num):
	if iters == 0:
		return None
	wins = 0
	for i in range(iters):
		if player_num == 1:
			env.play(agent, opponent, 1, silence = True)
		else:
			env.play(opponent, agent, 1, silence = True)
		if  env.is_over() == player_num:
			wins += 1
	print("Evaluation: {}% -- {} Wins / {} Games".format(100 * float(wins) / iters,
														 wins, iters))

def elapsed_time(start):
	new_time = time() - start
	elapsed = [int(i) for i in convert(new_time)]
	clock = localtime(time())[3:5]
	return "Elapsed {} : {} : {} (at {} : {})".format(elapsed[0], elapsed[1],
													 elapsed[2], clock[0],
													 clock[1])

def convert(seconds):
	minutes = seconds // 60
	return [minutes // 60, minutes % 60, seconds % 60]

VERSION = 13
print("Version {}".format(VERSION))
rewards = {
	"illegal": -100,
    "forced": 10,
    "loss": -5
}
e = Env(5, rewards = rewards)
a1 = Q([25, 100, 200, 500, 100, 25], gamma = .6, epsilon = 0.1, beta = 0.0,
		 name = "5x5_server_p1_{}".format(VERSION))
a2 = Q([25, 100, 200, 500, 100, 25], gamma = .6, epsilon = 0.1, beta = 0.0,
		 name = "5x5_server_p2_{}".format(VERSION))
t1 = Trainer(a1, learn_rate = 1e-12, record = False, change_agent_epsilon = True,
			 epsilon_func = lambda x: min(1.0, 10.0 / x))
t2 = Trainer(a2, learn_rate = 1e-12, record = False, change_agent_epsilon = True,
			 epsilon_func = lambda x: min(1.0, 10.0 / x))

train_agent(e, a1, a2, t1.get_episode(rotate = True),
			t2.get_episode(rotate = True))