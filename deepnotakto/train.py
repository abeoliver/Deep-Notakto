# train.py
# Abraham Oliver, 2017
# Deep Notakto Project

import environment as env
import trainer, os, sys
import agents.Q as Q
import agents.random_plus as random_plus

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
e = env.Env(4)
print("Initializing Players...")
e = env.Env(4)
p2 = Q.Q([16, 100, 100, 16], gamma = .3, epsilon = 0.1, beta = 3.0, name = "4x4")
p2_train = trainer.Trainer(p2, 1e-8)
p1 = random_plus.RandomAgentPlus()
print("Beginning Training...")
its = int(sys.argv[1])
for i in range(its):
	if i % (its / 10) == 0:
		print("{} / {}".format(i, its))
	e.play(p1, p2, 1, trainer_a2 = p2_train.get_episode(1e-7, rotate = True),
		   final_reward = True, silence = True)
print("Saving params...")
p2.save()
print("Done")
