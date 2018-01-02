# train.py
# Abraham Oliver, 2017
# Deep Notakto Project
from time import time

import deepnotakto.util as util

def train_agent(env, a1, a2, rounds, round_length = 1, save_a1 = False,
				save_a2 = False, path = "", record_name = "record.txt",
				constant = True):
	print("Traning Agents '{}' vs '{}'".format(a1.name, a2.name))
	start = time()
	if save_a1:
		util.record(record_name, a1)
	if save_a2:
		util.record(record_name, a2)
	r = 0
	while r < rounds or rounds < 0:
		wins = [0, 0]
		# Play a round, adding the wins to the count
		for game in range(round_length):
			if game % 2 == 0 or constant:
				wins[play(env, a1, a2) - 1] += 1
			else:
				# Winner is opposite of agent number
				wins[0 if play(env, a2, a1) == 2 else 1] += 1
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
		# Check for illegal move or a win
		if observation["info"]["illegal"] or observation["done"]:
			break
	# End the game
	a1.save_episode()
	a2.save_episode()
	# Return winner
	return env.turn % 2 + 1

def ef(a, b, c = 1):
	"""
	Epsilon change function
	Parameters:
	     a (int) - Number of iterations for 100%
	     b (int) - Number of iterations for 50%
	     c (float) - Steepness modifier
	"""
	if a < 0:
		raise ValueError("Argument 'a' (val {}) cannot be less than or equal to 0".format(a))
	if b < 0:
		raise ValueError("Argument 'b' (val {}) cannot be less than or equal to 0".format(b))
	if a >= b:
		raise ValueError("Argument 'a' (val {}) cannot be greater that or equal to argument 'b' (val {})".format(a, b))
	if c == 0:
		raise ValueError("Argument 'c' (val {}) cannot be 0".format(c))
	alpha, beta= pow(a, c), pow(b, c)
	q = lambda x: ((alpha * beta) + pow(x, c) * (beta - 2 * alpha)) / (2 * pow(x, c) * (beta - alpha))
	return lambda x: max(0, min(1, q(x))) if x != 0 else 0

def console_print(start, wins, round, rounds, round_length):
	pct = int((float(wins[0]) / round_length) * 100)
	print(util.elapsed_time(start))
	print("Score: {}% vs {}% out of {} games [{} / {} rounds played]\n".format(
		pct, 100 - pct, round_length, round + 1, rounds
	))