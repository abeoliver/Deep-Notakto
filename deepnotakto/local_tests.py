from datetime import datetime
from numpy import zeros
from deepnotakto.notakto import QTree, GuidedNode


def timer(func, *args, **kwargs):
    # Run function
    t = datetime.now()
    f = func(*args, **kwargs)
    nt = datetime.now() - t
    return [f, nt]


total = datetime.now() - datetime.now()
i = 10
for _ in range(i):
    agent = QTree(10, [])
    node = GuidedNode(zeros([10,10]), agent)
    total += timer(node.action_space)[1]
print("AVG {}".format(total / i))