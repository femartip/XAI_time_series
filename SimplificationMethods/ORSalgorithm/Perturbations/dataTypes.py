from typing import List

from ORSalgorithm.Utils.line import interpolate_points_to_line


class SegmentedTS:
    x_pivots: List[int]
    y_pivots: List[float]
    ts_length: int

    line_version: List[float]

    pred_class: int

    def __init__(self, x_pivots: List[int], y_pivots: List[float], ts_length: int):
        self.x_pivots = x_pivots
        self.y_pivots = y_pivots
        self.ts_length = ts_length
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
