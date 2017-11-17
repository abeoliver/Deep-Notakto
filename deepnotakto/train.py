# train.py
# Abraham Oliver, 2017
# Deep Notakto Project

import os
from time import time
from random import choice
import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_agent(env, a1, a2, rounds, round_length = 100, save_a1 = False,
				save_a2 = False, path = "", record_name = "record.txt"):
	print("Traning Agents '{}' vs '{}'".format(a1.name, a2.name))
	start = time()
	if save_a1:
		util.record(record_name, a1)
	if save_a2:
		util.record(record_name, a2)
	r = 0
	a1_e = lambda x: .5
	a2_e = lambda x: 0
	while r < rounds or rounds < 0:
		"""
		if r % 200 == 0:
			a1.change_param("epsilon_func", a1_e)
			a2.change_param("epsilon_func", a2_e)
			a1_e, a2_e = [a2_e, a1_e]"""
		wins = [0, 0]
		# Play a round, adding the wins to the count
		for _ in range(round_length):
			wins[play(env, a1, a2) - 1] += 1
		# Console information
		console_print(start, wins, r, rounds, round_length)
		if save_a1:
			util.update(record_name, a1, "{} / {}".format(wins[0], round_length))
			a1.save("{}{}.npz".format(path, a1.name))
		if save_a2:
			util.update(record_name, a2, "{} / {}".format(wins[1], round_length))
			a2.save("{}{}.npz".format(path, a2.name))
		# Incrememnt round counter
		r += 1

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