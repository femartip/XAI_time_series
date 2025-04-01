import numpy as np
from typing import Dict, Tuple, List
import random
from collections import defaultdict as D
from collections import namedtuple as T
import logging
import os

from ORSalgorithm.simplify.plotting import plot
from ORSalgorithm.simplify.MinHeap import MinHeap
from ORSalgorithm.Utils.types import HeapStruct
from ORSalgorithm.Utils.scoring_functions import score_closeness


Solution = T("Solution", "error currentIdx currentOrder prevIdx prevOrder last_seg")
Function = T("Function", "m b")
logging.basicConfig(level=logging.INFO)

def sol(error, currentIdx, currentOrder, prevIdx, prevOrder, last_seg):
    return Solution(error, currentIdx, currentOrder, prevIdx, prevOrder, last_seg)  # please never round


def heap(error: float, lineSegError: float, j: int, i: int, order: int, last_seg: bool):
    """

    :param error: Total error
    :param lineSegError: Error from line segment from j to i
    :param j: Start of line
    :param i: End of line
    :param order: Currently evaluating k-best option from [0,j].
    :param last_seg: Extrapolate the line to 0.
    :return:
    """
    return HeapStruct(error, lineSegError, j, i, order, last_seg)


sol0 = sol(0, 0, 0, 0, 0, True)
VINF = sol(float("inf"), 0, 0, 0, 0, True)


def _line_error(f: Function, X, Y, alpha, ts_length) -> float:
    ts1 = [f.m * x + f.b for x in X]
    ts2 = Y
    error = score_closeness(ts1=ts1, ts2=ts2, alpha=alpha, ts_length=ts_length)
    return error
    # error = 0
    # for x, y in zip(X, Y):
    #    error += abs(f.m * x + f.b - y) ** 2
    # return alpha * math.sqrt(error) / distance_weight


def _gen_line(x1, y1, x2, y2) -> Function:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return Function(m, b)


def segmented_least_squares_DP(X: List[int], Y: List[float] | np.ndarray, K: int,
                               alpha: float, beta:float) -> \
        Dict[
            Tuple[int, int], Solution]:
    """
    Least squares solution using segmented least squares DP algo.
    Parameters
    @ X:param, List of X values
    @ Y:param, List of Y values
    @ c:param, Punishment for more segments
    @ K:param, Top K best solutions
    @ distance_weight:param, Weight of the distance
    @ alpha:param, ?????
    """
    logging.debug(f"X:{X}")
    logging.debug(f"Y:{Y}")
    OPT = D(lambda: VINF)
    # Base case for only one point
    OPT[0, 0] = sol0

    ts_length = len(X)
    segment_punishment = beta*(1/(ts_length-1))
    # Solve DP
    for i in range(1, len(X)):
        min_heap_top_k_solutions = MinHeap()  # Size: O(2*i)
        for j in range(0, i):
            f = _gen_line(X[j], Y[j], X[i], Y[i])

            # Keep other segments
            f_line_error = _line_error(f, X[j:i + 1], Y[j:i + 1], alpha=alpha, ts_length=ts_length)
            f_segment_keep_error = OPT[j, 0].error + segment_punishment + f_line_error

            heap_best_opt_j = heap(f_segment_keep_error, f_line_error, j, i, 0, last_seg=False)
            min_heap_top_k_solutions.insert(heap_best_opt_j)

            if j != 0:
                # Draw a line to the start
                f_total_line_error = _line_error(f, X[0:i + 1], Y[0:i + 1], alpha=alpha, ts_length=ts_length)
                f_total_error = f_total_line_error + segment_punishment

                heap_final_best_opt_j = heap(f_total_error, None, j, i, 0, last_seg=True)
                min_heap_top_k_solutions.insert(heap_final_best_opt_j)

        # For each possible j we have now found the best solution, let's search for the overall k best.
        if i == len(X) - 1:
            logging.debug("Stop!")
        idx_k = 0
        while idx_k < K and min_heap_top_k_solutions.getMin() is not None:
            heapObj = min_heap_top_k_solutions.removeMin()
            if heapObj.last_seg:
                k_best = sol(error=heapObj.error, currentIdx=i, currentOrder=idx_k, prevIdx=heapObj.j,
                             prevOrder=heapObj.order, last_seg=True)
                OPT[i, idx_k] = k_best
            else:
                k_best = sol(error=heapObj.error, currentIdx=i, currentOrder=idx_k, prevIdx=heapObj.j,
                             prevOrder=heapObj.order, last_seg=False)
                OPT[i, idx_k] = k_best

                # Add the next best option from j to the heap
                if OPT[heapObj.j, heapObj.order + 1].error < float("inf"):
                    new_total_error = OPT[heapObj.j, heapObj.order + 1].error + segment_punishment + heapObj.lineSegError
                    newHeapObj = heap(error=new_total_error, lineSegError=heapObj.lineSegError, j=heapObj.j, i=i,
                                      order=heapObj.order + 1, last_seg=False)
                    min_heap_top_k_solutions.insert(newHeapObj)
            idx_k += 1

    # Currently OPT[n-1] holds the best solution, if we require the algorithm to pick n, lets fix this.
    # Let's manually check all possible last segments, going through two points in the TS.
    # We will leave the DP for the area before last segment.
    min_heap_last_point = MinHeap()  # Size: O(2*i)
    for i in range(1, len(X)):
        for j in range(0, i):
            # Line between i and j
            f = _gen_line(X[j], Y[j], X[i], Y[i])

            # Find error j to end.
            f_line_out_error = _line_error(f, X[j:], Y[j:], alpha=alpha, ts_length=ts_length)
            f_out_error = OPT[j, 0].error + segment_punishment + f_line_out_error  # OPTIMAL 0..j + c + rest
            heap_best_opt_j = heap(f_out_error, f_line_out_error, j, i, 0, last_seg=False)
            min_heap_last_point.insert(heap_best_opt_j)

            if j != 0:  # Don't want duplicate
                # Find error 0 to end
                f_line_full_error = _line_error(f, X, Y, alpha=alpha, ts_length=ts_length)
                f_out_full_error = beta + f_line_full_error  # c + ALL
                heap_best_all_opt_j = heap(f_out_full_error, None, j, i, 0, last_seg=True)
                min_heap_last_point.insert(heap_best_all_opt_j)

    idx_k = 0
    while idx_k < K and min_heap_last_point.getMin() is not None:
        last_heapObj = min_heap_last_point.removeMin()
        if last_heapObj.last_seg:
            k_best = sol(error=last_heapObj.error, currentIdx=last_heapObj.i, currentOrder=idx_k,
                         prevIdx=last_heapObj.j,
                         prevOrder=last_heapObj.order, last_seg=True)
            OPT[len(X) - 1, idx_k] = k_best
            if idx_k == 0:
                logging.debug("Error check", k_best.error, k_best.currentIdx, k_best.prevIdx)
        else:
            k_best = sol(error=last_heapObj.error, currentIdx=last_heapObj.i, currentOrder=idx_k,
                         prevIdx=last_heapObj.j,
                         prevOrder=last_heapObj.order, last_seg=False)
            OPT[len(X) - 1, idx_k] = k_best
            if idx_k == 0:
                logging.debug("Error check", k_best.error, k_best.currentIdx, k_best.prevIdx)

            # Add the next best option from j to the heap
            if OPT[last_heapObj.j, last_heapObj.order + 1].error < float("inf"):
                new_total_error = OPT[last_heapObj.j, last_heapObj.order + 1].error + beta + last_heapObj.lineSegError
                newHeapObj = heap(error=new_total_error, lineSegError=last_heapObj.lineSegError, j=last_heapObj.j,
                                  i=last_heapObj.i,
                                  order=last_heapObj.order + 1, last_seg=False)
                min_heap_last_point.insert(newHeapObj)
        idx_k += 1
    return OPT


def solve(X, Y, K, alpha: float, beta:float) -> Dict[Tuple[int, int], Solution]:
    OPT = segmented_least_squares_DP(X, Y, K, alpha=alpha, beta=beta)
    return OPT


def extract_points(OPT: Dict[Tuple[int, int], Solution], k: int, X):
    """
    :param OPT: The OPT dict
    :param k: k-th best solution
    :param X: timeseries
    :return: k-best approximation
    """
    solution = OPT[len(X) - 1, k]
    list_of_points_last_first = [solution.currentIdx]
    while (not solution.last_seg) and (solution.currentIdx != solution.prevIdx):
        solution = OPT[solution.prevIdx, solution.prevOrder]
        list_of_points_last_first.append(solution.currentIdx)

    if solution.last_seg and solution.currentIdx != solution.prevIdx:
        list_of_points_last_first.append(solution.prevIdx)

    return list(reversed(list_of_points_last_first))


def solve_and_find_points(X, Y, K, alpha: float, beta:float, saveImg=False):
    """

    :param X: X_values in timeseries
    :param Y: Y_values in timeseries
    :param c: Punishment for more segments
    :param K: Top k best solution
    :param alpha: ???
    :param saveImg:
    :return: all_selected_points, all_ys
    """
    #assert beta == 1/ (len(X)-1) # normalize seg coun
    #beta = 1/ (len(X)-1) # normalize seg count
    logging.debug("Solve done")
    OPT = solve(X, Y, K, alpha=alpha, beta=beta)
    if saveImg:
        logging.debug("Making images...")
    logging.debug("Min error:", OPT[len(X) - 1, 0].error)
    logging.debug("Max error:", OPT[len(X) - 1, K - 1].error)
    all_selected_points = []
    all_ys = []
    for k in range(K):
        selected_points = extract_points(OPT, k, X)
        ys = [Y[i] for i in selected_points]
        if saveImg:
            logging.debug(f"{k}/{K}")
            plot(X, Y, selected_points, ys, fname=f"simplify/img/{k}")

        all_selected_points.append(selected_points)
        all_ys.append(ys)
    return all_selected_points, all_ys


if __name__ == "__main__":
    random.seed(6)
    ts_x = list(range(20))
    ts_y = [random.randint(-10, 10) for _ in range(len(ts_x))]
    my_c = 1
    my_k = 10000
    weight = max(ts_y) - min(ts_y)
    solve_and_find_points(ts_x, ts_y, my_k, weight)
