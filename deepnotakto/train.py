# train.py
# Abraham Oliver, 2017
# Deep Notakto Project

import environment as env
import trainer, util
import agents.activated as activated
import agents.Q as Q
import agents.random_plus as random_plus

e = env.Env(4)
print("Initializing Players...")
p2 = Q.Q([16, 100, 100, 16], gamma = .3, epsilon = 0.1, beta = 3.0, name = "4x4")
p2_train = trainer.Trainer(p1, 1e-8)
p1 = random_plus.RandomAgentPlus()
print("Beginning Training...")
e.play(p1, p2, 20000, trainer_a2 = trainer.get_episode(1e-7, rotate = True),
       final_reward = True)
print("Done")