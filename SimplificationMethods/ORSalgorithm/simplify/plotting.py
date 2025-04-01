from collections import defaultdict as D
from collections import namedtuple as T
from typing import Dict

Solution = T("Solution", "error index prev last_seg")
Function = T("Function", "m b")


def _gen_line(x1, y1, x2, y2) -> Function:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return Function(m, b)


def _eval(f, x):
    """Evaluates a function f = ax + b on point x"""
    a, b = f
    return a * x + b


def _fit_points(x1, x2, px1, py1, px2, py2):
    """Generate the line between p1 and p2"""
    f = _gen_line(px1, py1, px2, py2)
    return _eval(f, x1), _eval(f, x2)


def plot(X, Y, pX, pY, fname="Plot"):
    import matplotlib.pyplot as plt
    plt.clf()
    XMIN = -1
    XMAX = len(X) + 1
    YMIN = min(Y) - 5
    YMAX = max(Y) + 5
    plt.xlim(XMIN, XMAX)
    plt.ylim(YMIN, YMAX)

    plt.plot(X, Y, "o", markersize=3, c="black")

    if pX[0] != X[0]:
        plt.plot((X[0], pX[0]), _fit_points(X[0], pX[0], pX[0], pY[0], pX[1], pY[1]), "--", markersize=3, c="red")

    for i in range(len(pX) - 1):
        plt.plot((pX[i], pX[i + 1]), (pY[i], pY[i + 1]), "--", markersize=3, c="red")

    if pX[-1] != X[-1]:
        plt.plot((pX[-1], X[-1]), _fit_points(pX[-1], X[-1], pX[-2], pY[-2], pX[-1], pY[-1]), "--", markersize=3,
                 c="red")

    plt.savefig(f"{fname}.png")
