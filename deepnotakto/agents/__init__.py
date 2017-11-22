# agents/__init__.py
# Abraham Oliver, 2017
# Deep-Notakto Project

from deepnotakto.agents.agent import Agent
from deepnotakto.agents.human import Human
from deepnotakto.agents.random_agent import RandomAgent
from deepnotakto.agents.Q import Q
import deepnotakto.agents.activated

__all__ = dir()