import sys

firsts = []
bests = []

for i_test in range(100):
    with open("{}{}.mcts".format(sys.argv[1], i_test)) as f:
        first = None
        best = -1
        for i, line in enumerate(f):
            try:
                a = float(line.split(' ')[0])
            except:
                continue

            if a > 0 and first is None:
                first = i

            if a > best:
                best = a

    firsts.append(first)
    bests.append(best)

import numpy as np

print(firsts)
print(bests)
print(np.mean(firsts))
print(np.mean(bests))
