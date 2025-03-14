import random
import numpy as np


def generate(l, k, n):
    X = np.array(range(n))
    _yarr = list(range(1, k + 1))
    _act_y = []
    for i in range(l):
        tmp = []
        offset = random.randint(1, 5)
        for e in _yarr:
            tmp.append(random.gauss(offset + e, 0.8))
        if random.random() > 0.5:
            tmp = reversed(tmp)
        _act_y += tmp
    Y = np.array(_act_y)
    for i in range(n):
        x, y = X[i], round(Y[i], 5)
        print(f"{x},{y}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        exit("Usage: generator l k, e.g. generator 3 5")
    L = int(sys.argv[1])
    k_ = int(sys.argv[2])
    N = L * k_
    generate(L, k_, N)
