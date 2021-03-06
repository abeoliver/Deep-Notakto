#######################################################################
#  Can Deep Reinforcement Learning Solve Misère Combinatorial Games?  #
#  File: agents/__init__.py                                           #
#  Abraham Oliver, 2018                                               #
#######################################################################

from deepnotakto.agents.agent import Agent
from deepnotakto.agents.human import Human
from deepnotakto.agents.Q import Q, QTrainer
from deepnotakto.agents.qtree import QTree, QTreeTrainer

__all__ = dir()
