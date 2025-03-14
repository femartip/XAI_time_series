def _eval(f, x):
    """Evaluates a function f = ax + b on point x"""
    a, b = f
    return a * x + b


def _fit_points(f, x1, x2):
    """Simply evaluate two points and returns the tuple"""
    return _eval(f, x1), _eval(f, x2)


def plot(OPT, X, Y, L):
    import matplotlib.pyplot as plt

    XMIN = -1
    XMAX = len(X) + 1
    YMIN = -5
    YMAX = max(Y) + 5
    l = L
    i = len(X)
    opt = OPT[l, i]
    plt.xlim(XMIN, XMAX)
    plt.ylim(YMIN, YMAX)
    _opt_str = f"{round(opt.opt,1)}"
    label = f"L={l}, N={i}, opt={_opt_str.ljust(8)}"

    # Plot the original points
    plt.plot(X[:i], Y[:i], "o", markersize=3, c="black", label=label)
    plt.legend(loc="upper right")

    # Plot slopes (going from right to left)
    while opt.l > 0:
        f = opt.slope
        x1 = opt.pre
        x2 = opt.i - 1
        plt.plot((x1, x2), _fit_points(f, x1, x2))

        opt = OPT[opt.l - 1, opt.pre]

    plt.show()
