# train.py
# Abraham Oliver, 2017
# Deep Notakto Project

import environment as env
import trainer, util
import agents.activated as activated
import agents.Q as Q
import agents.random_plus as random_plus

e = env.Env(3)
print("Initializing Players")
p1 = activated.SigmoidHidden([9, 100, 200, 100, 9], gamma = .6,
                              epsilon = .1, beta = 5.0)
p1_train = trainer.Trainer(p1, 1e-5)
p2 = random_plus.RandomAgentPlus()
for i in range(10000):
    e.play(p1, p2, 1000, trainer_a1 = p1_train.get_online(1e-6))
    p1_train.offline(p1.states, p1.actions, p1.rewards,
                     epochs = 10, batch_size = 20)
    p1.save()
    p1.reset_memory()
print("Done")