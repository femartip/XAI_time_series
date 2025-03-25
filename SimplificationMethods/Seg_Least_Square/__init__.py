import sys
import csv
from .generator import generate
from .segmentedls import solve
from .plotting import plot
from Utils.dataTypes import SegmentedTS

"""
From https://github.com/pgdr/seglines
"""


def convert_to_segmentedTS(segments:list, ts_length) -> SegmentedTS:
    pivot_x = []
    pivot_y = []

    num_segments = len(segments)

    segments = list(reversed(segments))

    for segment in segments:
        start_x = segment[0][0]
        start_y = segment[0][1]
        end_x = segment[1][0]
        end_y = segment[1][1]
        pivot_x.append(start_x)
        pivot_y.append(start_y)

        if start_x != end_x:
            pivot_x.append(end_x)
            pivot_y.append(end_y)

    return SegmentedTS(pivot_x, pivot_y, ts_length=ts_length, num_real_segments=num_segments)


def _slope_to_str(slope):
    a, b = slope
    return f"f(x) = {a:.3f}·x + {b:.3f}"


def run(X, Y, L, do_plot=False, do_print=False):
    OPT = solve(X, Y, L)

    N = len(X)

    l = L
    i = len(X)

    opt = OPT[l, i]

    segments = []
    result = []

    digits_ = len(str(i))

    while opt.l > 0:
        x1, y1 = opt.pre, Y[max(0, opt.pre)]
        x2, y2 = opt.i - 1, Y[max(0, opt.i - 1)]
        segments.append(((x1, round(y1, 2)), (x2, round(y2, 2)), opt.slope))
        opt = OPT[opt.l - 1, opt.pre]

    if do_print:
        for idx, (start, end, slope) in enumerate(reversed(segments)):
            s, s_val = start
            e, e_val = end
            slope_str = _slope_to_str(slope)
            print(f"segment {idx+1:2}: ", end="")
            print(f"{s:{digits_}} ({s_val:.3f})".ljust(digits_ + 13), end="")
            print(f"{e:{digits_}} ({e_val:.3f})".ljust(digits_ + 13), end="")
            print(slope_str)

    if do_plot:
        plot(OPT, X, Y, L)

    return convert_to_segmentedTS(segments, N)


