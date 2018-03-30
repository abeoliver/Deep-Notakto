from datetime import datetime
from numpy import zeros
from deepnotakto.agents.qtree import QTree
from deepnotakto.treesearch import GuidedNotaktoNode


def timer(func, *args, **kwargs):
    # Run function
    t = datetime.now()
    f = func(*args, **kwargs)
    nt = datetime.now() - t
    return [f, nt]


total = datetime.now() - datetime.now()
i = 10
for _ in range(i):
    agent = QTree(3, [])
    node = GuidedNotaktoNode(zeros([3,3]), agent)
    total += timer(node.action_space)[1]
    print()
print("AVG {}".format(total / i))