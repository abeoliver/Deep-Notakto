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
	for i in range(episodes):
		if i % (episodes / 100) == 0 and int((i / episodes) * 100) % PRINT_FREQ_PERCENT == 0:
			print("{}% [{} / {}] {}".format(
				int((i / episodes) * 100), i, episodes, elapsed_time(start)))
		env.play(p1, p2, 1, trainer_a1 = t1, trainer_a2 = t2,
				 final_reward = True, silence = True)
		if i % SAVE_FREQ == 0:
			if t1 != None:
				p1.save(name="/scratch/abraoliv/{}.npz".format(p1.name))
			if t2 != None:
				p2.save(name="/scratch/abraoliv/{}.npz".format(p2.name))
		if i % EVAL_FREQ == 0:
			evaluate(env, p1, p2, int(i / 10000), 1)
	print("100% [{} / {}]  Elapsed : {}".format(episodes, episodes, elapsed_time(start)))
	evaluate(env, p1, p2, 1, 1)
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

TRAIN_REVS = 10000
PRINT_FREQ_PERCENT = 10
SAVE_FREQ = 100
EVAL_FREQ = 10000
VERSION = 17
print("Version {}".format(VERSION))
rewards = {
	"ilegal": -100,
    "forced": 10,
    "loss": -5
}

for i in range(3, 6):
	if i == 3 or i == 5:
		player = 1
	elif i == 4:
		player = 2
	e = Env(i, rewards = rewards)
	a = Q([i ** 2, 100, 200, i ** 2], gamma = .6, beta = 0.0,
			 name = "{}x{}_server_p{}_{}".format(i, i, player, VERSION))
	t = Trainer(a, learn_rate = 1e-8, record = False, change_agent_epsilon = True,
				 epsilon_func = lambda x: min(1.0, 1000.0 / x))
	rand = RandomAgentPlus()
	if player == 1:
		train_agent(e, 100000, a, rand, t1 = t)
	else:
		train_agent(e, 100000, rand, a, t2 = t)
	wins = 0
	for i in range(100):
		if player == 1:
			e.play(a, rand)
		else:
			e.play(rand, a)
		over = e.is_over()
		if over == player:
			wins += 1
	util.record("all_trials.txt", a, t, wins)
