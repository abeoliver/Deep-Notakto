import agents.Q as Q
from agents.human import Human
from environment import Env
from visual import GameWithConfidences
from agents.random_agent import RandomAgent

e = Env(3)
a1 = Q.load_agent("agents/saves/Q_relu_hidden(0bb08).npz")
# a2 = RandomAgent(e)
a2 = Human()
vis = GameWithConfidences(e, a1, a2, show_confidences = True)
