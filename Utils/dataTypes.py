from typing import List

def calculate_line_equation(x1, y1, x2, y2, x3):
    # Calculate the slope (m)
    delta_x = x2 - x1
    if delta_x == 0:
        raise ValueError("The points must have different x-coordinates to calculate the slope.")
    m = (y2 - y1) / delta_x

    # Calculate the y-intercept (b)
    b = y1 - m * x1

    # Calculate y3 at x3
    y3 = m * x3 + b

    return y3

def interpolate_points_to_line(ts_length: int, x_selected: List[int], y_selected: List[float]) -> List[float]:
    """
    Given a list (points) of [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] of selected points calculate the y value of
    each timeStep.

    For each x in range(timeStep) we have 3 cases:
    1. x1 <= x <= x4: Find the pair xi <= x <=xi+1, s.t. i<=3. Use this slope to find the corresponding y value.
    2. x < x1. Extend the slope between x1 and x2 to x, and find the corresponding y value.
    3. x4 < x. Extend the slope between x3 and x4 to x, and find the corresponding y value.
    :param ts_length:
    :param y_selected:
    :param x_selected:
    :return:
    """

    interpolation_ts = [0 for _ in range(ts_length)]
    pointsX = 0
    for x in range(ts_length):
        # If x is bigger than x_selected[pointsX+1] we are in the next interval
        # pointsX < len(x_selected) - 2 Indicates that we extrapolate the two last points even if x is after this.
        if x > x_selected[pointsX + 1] and pointsX < len(x_selected) - 2:
            pointsX += 1

        x1 = x_selected[pointsX]
        x2 = x_selected[pointsX + 1]
        y1 = y_selected[pointsX]
        y2 = y_selected[pointsX + 1]
        x3 = x
        y3 = calculate_line_equation(x1, y1, x2, y2, x3)
        interpolation_ts[x] = y3

    return interpolation_ts

class SegmentedTS:
    x_pivots: List[int]
    y_pivots: List[float]
    ts_length: int
    num_real_segments: int = None #Only used if segments are non contiguous

    line_version: List[float]

    pred_class: int

    def __init__(self, x_pivots: List[int], y_pivots: List[float], ts_length: int, num_real_segments: int = None):
        self.x_pivots = x_pivots
        self.y_pivots = y_pivots
        self.ts_length = ts_length
        self.num_real_segments = num_real_segments
        self.set_line_version(ts_length)

    def set_class(self, pred_class: int):
        self.pred_class = pred_class

    def set_line_version(self, ts_length: int):
        line_version = interpolate_points_to_line(ts_length=ts_length, x_selected=self.x_pivots,
                                                  y_selected=self.y_pivots)
        self.line_version = line_version

    def __repr__(self):
        return "--".join([f"({x},{y})" for x,y in zip(self.x_pivots, self.y_pivots)])


class SinglePointPerturbation:
    new_x: int
    new_y: float
    idx_pivots: int

    perturbationTS: SegmentedTS

    def __init__(self, new_x: int, new_y: float, idx_pivots: int, x_pivots: List[int], y_pivots: List[float],
                 ts_length: int):
        self.perturbationTS = SegmentedTS(x_pivots=x_pivots, y_pivots=y_pivots, ts_length=ts_length)

        self.new_x = new_x
        self.new_y = new_y
        self.idx_pivots = idx_pivots

    def set_line_version(self, ts_length: int):
        self.perturbationTS.set_line_version(ts_length=ts_length)

    def set_class(self, pred_class: int):
        self.perturbationTS.set_class(pred_class=pred_class)
