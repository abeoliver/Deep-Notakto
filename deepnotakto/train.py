# train.py
# Abraham Oliver, 2017
# Deep Notakto Project

from time import time

import os

from time import time, localtime
from pickle import dump, load
from numpy import sqrt

from deepnotakto.util import load_agent, average_value, seconds_to_time
from deepnotakto.agents.random_agent import RandomAgent
from deepnotakto.environment import Env
from deepnotakto.agents.qtree import QTree
from deepnotakto.statistics import measure

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
	if observation["info"]["illegal"]:
		return 2 - (env.turn % 2)
	return env.turn % 2 + 1

def tournament(env, a1, a2, games):
	wins = [0, 0]
	for _ in range(games):
		wins[play(env, a1, a2) - 1] += 1
	return wins

def elapsed_time(start):
	new_time = time() - start
	elapsed = [int(i) for i in seconds_to_time(new_time)]
	clock = localtime(time())[3:5]
	return "Time                  {} : {} : {} (at {} : {} {})".format(
        elapsed[0], str(elapsed[1]).zfill(2), str(elapsed[2]).zfill(2),
        int(clock[0]) % 12 if int(clock[0]) not in [0, 12] else 12, str(clock[1]).zfill(2),
        "PM" if clock[0] >= 12 and int(clock[0]) != 0 else "AM")

def train_model_with_tournament_evaluation(agent, statistics, model_path, stats_path,
										   best_model_path = None, save_every = 1,
										   sims = 100, player = 1, games = 100):
	""" Run a training loop with evaluation against a random opponent """
	# Clean input
	if best_model_path == None:
		best_model_path = model_path + "_best"
	# Environment to play in
	env = Env(agent.size)
	# Opponent to play against
	opponent = RandomAgent(env)

	# Value for comparing best model and current model
	prev_best_model_val = -1
	# Current best model
	current_best = opponent

	# Begin console output with save location and time
	clock = localtime(time())[3:5]
	o = "\n\n-------- {} --------\nSaved as '{}'\nStarted at {} : {} {}\n".format(
		agent.name, model_path, int(clock[0]) % 12 if clock[0] not in [0, 12] else 12,
		str(clock[1]).zfill(2), "PM" if clock[0] >= 12 and int(clock[0]) != 0 else "AM")
	print(o)

	# Start clock
	start = time()

	# Main training loop
	while True:
		# Run self play training algorithm
		print("Self play... ", end = "")
		agent.self_play(games = save_every, simulations = sims)
		print("Completed")
		# Save the model
		agent.save(model_path)
		# Prepare console output
		outputs = []
		outputs.append("TIME PLACEHOLDER")
		outputs.append("Iteration             {}".format(agent.iteration))

		# Q-based evaluation tournament
		agent.mode("q")
		print("Q-based evaluation... ", end = "")
		if player == 1:
			q_wins = tournament(env, agent, opponent, games)[0]
		else:
			q_wins = tournament(env, opponent, agent, games)[1]
		print("Complete")
		outputs.append("Q Evaluation          {}%".format(int(q_wins * 100 / games)))

		# Best Tournament
		print("Against current best evaluation... ", end = "")
		best_wins_1 = tournament(env, agent, current_best, games // 2)[0]
		best_wins_2 = tournament(env, current_best, agent, games // 2)[1]
		print("Complete")
		outputs.append("Best Evaluation       {}%".format(
			int((best_wins_1 + best_wins_2) * 100 / games)))

		# Designate as best model if is better than previous model
		is_best = False
		if (q_wins / games) > prev_best_model_val:
			prev_best_model_val = q_wins / games
			agent.save(best_model_path)
			is_best = True
			current_best = agent
			outputs.append("BEST MODEL")

		# Save statistics
		statistics[agent.iteration] = measure(agent, q_wins = q_wins, time=time() - start,
											  best = is_best, best_wins_1 = best_wins_1,
											  best_wins_2 = best_wins_2, best_model_val = prev_best_model_val)

		# Fill in placeholders
		outputs[0] = elapsed_time(start)
		# Construct final output string
		output = "".join(i + "\n" for i in outputs)
		print(output)
		# Save statistics
		with open(stats_path, "wb") as f:
			dump(statistics, f)

def competitive_training():
	pass