from typing import List
import numpy as np
from ORSalgorithm.Perturbations.dataTypes import SegmentedTS


def score_closeness(ts1: List[float], ts2: List[float] | np.ndarray, distance_weight: float,
                    alpha: float) -> float:
    # This should use the same function as the DP algo!!
    error = 0
    for y1, y2 in zip(ts1, ts2):
        error += abs(y1 - y2) ** 2
    return error * alpha / distance_weight


def score_simplicity(approximation: SegmentedTS, beta: float):
    simplicity = (len(approximation.x_pivots) - 1) * beta
    return simplicity
