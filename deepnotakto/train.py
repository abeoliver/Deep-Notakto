# train.py
# Abraham Oliver, 2017
# Deep Notakto Project

import os
from environment import Env
from trainer import Trainer
from agents.Q import Q
from agents.random_agent import RandomAgent
from time import time
from random import choice
import util
import agents.activated as activated

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_agent(env, a1, a2, rounds, round_length = 100, save_a1 = True,
				save_a2 = False):
	start = time()
	if save_a1:
		util.record("BB.txt", a1)
	if save_a2:
		util.record("BB2.txt", a2)
	for r in range(rounds):
		wins = [0, 0]
		# Play a round, adding the wins to the count
		for _ in range(round_length):
			wins[play(env, a1, a2) - 1] += 1
		# Console information
		console_print(start, wins, r, rounds, round_length)
		if save_a1:
			util.update("BB.txt", a1, "{} / {}".format(wins[0], round_length))
			a1.save("agents/saves/{}.npz".format(a1.name))
		if save_a2:
			util.update("BB2.txt", a2, "{} / {}".format(wins[1], round_length))
			a2.save("agents/saves/{}.npz".format(a2.name))

def console_print(start, wins, round, rounds, round_length):
	pct = int((float(wins[0]) / round_length) * 100)
	print(util.elapsed_time(start))
	print("Score: {}% vs {}% out of {} games [{} / {} rounds played]\n".format(
		pct, 100 - pct, round_length, round + 1, rounds
	))

def play(env, a1, a2):
	a1.new_episode()
	a2.new_episode()
	# Reset environment
	env.reset()
	while True:
		# Get current player
		player = [a1, a2][env.turn % 2]
		# Play
		observation = player.act(env)
		# Update turn counter
		env.turn += 1
		# Check for illegal move or a win
		if observation["info"]["illegal"] or observation["done"]:
			break
	# End the game
	a1.save_episode()
	a2.save_episode()
	# Return winner
	return env.turn % 2 + 1

def ef(i, a = 1, b = 1, c = 0):
	return max(0, min(1, (float(a) / pow(i, b)) + c))

if __name__ == "__main__":
	util.new_record_file("BB2.txt")
	e = Env(4)
	train_dict = {"type": "episodic", "learn_rate": 1e-8, "rotate": True,
					   "epsilon_func": lambda x: ef(x, 300, .78, -0.08), "epochs": 1,
					   "batch_size": -1}
	"""
	a = Q([16, 100, 400, 16], gamma = .7, epsilon = 1.0,
		   training = train_dict) """
	a = util.load_agent("agents/saves/Q(baad4).npz", Q)
	a.training_params(training = train_dict)
	a.classifier = "a1a1a"
	a.name = "Q(a1a1a)"
	r = RandomAgent(e)
	train_agent(e, r, a, 100000, 100, save_a1 = False, save_a2 = True)