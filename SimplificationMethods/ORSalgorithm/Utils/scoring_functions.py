from typing import List
import numpy as np
from ..Perturbations.dataTypes import SegmentedTS


def score_closeness(ts1: List[float], ts2: List[float] | np.ndarray,
                    alpha: float, ts_length:int) -> float:
    # This should use the same function as the DP algo!!
    error = 0
    for y1, y2 in zip(ts1, ts2):
        error += abs(y1 - y2)
    return alpha*(error/ts_length)
     


def score_simplicity(approximation: SegmentedTS) -> float:
    simplicity = (len(approximation.x_pivots) - 1)  * (1 / (len(approximation.line_version) - 1))
    return simplicity
