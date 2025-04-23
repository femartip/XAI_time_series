import numpy as np
from matplotlib import pyplot as plt
from rdp import rdp
import logging
import time

from Utils.dataTypes import SegmentedTS
from SimplificationMethods.BottumUp.bottomUp import  get_swab_approx
from SimplificationMethods.Visvalingam_whyattt.Visvalingam_Whyatt import  simplify as VW_simplify
import SimplificationMethods.Seg_Least_Square as LS
from SimplificationMethods.ORSalgorithm.ORS_algorithm import get_simplifications

def get_OS_simplification(time_series: np.ndarray, alpha: float) -> list[SegmentedTS]:
    """
    Apply OS algorithm to simplify all time series in the dataset.
    """
    ts_simplifications = []
    first = False
    logging.debug(alpha)
    for ts_y in time_series:
        ts_x = [i for i in range(len(ts_y))]
        my_k = 1
        beta = 1/(len(ts_x)-1)*(1-alpha)
        init_time = time.time()
        selcted_xs, selectd_ys = get_simplifications(X=ts_x, Y=ts_y, K=my_k, alpha=alpha, beta=beta)
        logging.debug(f"Time: {(time.time()-init_time)}")
        if first:
            plt.figure()
            plt.plot(ts_x, ts_y, label="Original")
            plt.plot(selcted_xs[0], selectd_ys[0], label="Simplified")
            plt.plot(ts_x, SegmentedTS(x_pivots=selcted_xs[0], y_pivots=selectd_ys[0], ts_length=len(ts_y)).line_version, label="Line Simplified", linestyle="--")
            plt.legend()
            plt.show()
       
            first = False
        segTS = SegmentedTS(x_pivots=selcted_xs[0], y_pivots=selectd_ys[0], ts_length=len(ts_y))
        ts_simplifications.append(segTS)
    
    return ts_simplifications


def get_RDP_simplification(time_series: np.ndarray, epsilon: float) -> list[SegmentedTS]:
    """
    Apply Ramer-Douglas-Peucker (RDP) algorithm to simplify all time series in the dataset.
    """
    ts_simplifications = []
    first = False
    epsilon = epsilon
    logging.debug("epis:", epsilon)
    for ts_y in time_series:
        ts_x = list(range(len(ts_y)))
        init_time = time.time()
        simp_mask = rdp(np.array(list(zip(ts_x,ts_y))), epsilon=epsilon, return_mask=True)  # type: ignore
        logging.debug(f"Time: {(time.time()-init_time)}")
        simp_y = [y for y,bool in zip(ts_y, simp_mask) if bool]
        simp_x = [x for x,bool in zip(ts_x, simp_mask) if bool]
        if first:
            plt.figure()
            plt.plot(ts_x, ts_y, label="Original")
            plt.plot(simp_x, simp_y, label="Simplified")
            plt.plot(ts_x, SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)).line_version, label="Line Simplified", linestyle="--")
            plt.show()
            first = False
        ts_simplifications.append(SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)))
    
    return ts_simplifications


def get_bottom_up_simplification(time_series: np.ndarray, max_error: float, interpolate_segments: bool) -> list[SegmentedTS]:
    """
    Apply Bottom Up algorithm to simplify all time series in the dataset.
    """
    ts_simplifications = []
    first = False
    max_error = max_error
    logging.debug("max_error:", max_error)
    for ts_y in time_series:
        ts_x = list(range(len(ts_y)))
        simplification = get_swab_approx(ts_y, max_error=max_error)
        simp_x = simplification.x_pivots
        simp_y = simplification.y_pivots
        num_segments = simplification.num_real_segments
        
        if first:
            plt.figure()
            plt.title("Simplification using SWAB")
            plt.plot(ts_x, ts_y, label="Original")
            plt.plot(simp_x, simp_y, label="Simplified")
            plt.plot(ts_x, SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)).line_version,
                     label="Line Simplified", linestyle="--")
            plt.show()
            first = False
        if interpolate_segments:
            ts_simplifications.append(SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y), num_real_segments=num_segments))
        else:
            ts_simplifications.append(SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)))

    return ts_simplifications

def get_VW_simplification(time_series: np.ndarray, alpha: float) -> list[SegmentedTS]:
    """
    Apply Visvalingam Whyatt algorithm to simplify all time series in the dataset.
    """
    ts_simplifications = []
    first = False
    alpha = alpha
    logging.debug("alpha:", alpha)
    for ts_y in time_series:
        ts_x = list(range(len(ts_y)))
        simplification = VW_simplify(ts_y, alpha=alpha)
        simp_x = simplification.x_pivots
        simp_y = simplification.y_pivots
        if first:
            plt.figure()
            plt.title("Simplification using VC")
            plt.plot(ts_x, ts_y, label="Original")
            plt.plot(simp_x, simp_y, label="Simplified")
            plt.plot(ts_x, SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)).line_version,
                     label="Line Simplified", linestyle="--")
            plt.show()
            first = False
        ts_simplifications.append(SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)))

    return ts_simplifications


def get_LSF_simplification(time_series: np.ndarray, alpha: float) -> list[SegmentedTS]:
    """
    Apply Segmented Least Square Fit algorithm to simplify all time series in the dataset.
    """
    ts_simplifications = []
    first = False
    L = round(alpha * (len(time_series[0])))
    if L == 0: L = 1
    logging.debug("alpha:", alpha)
    for ts_y in time_series:
        ts_x = list(range(len(ts_y)))
        simplification = LS.run(ts_x, ts_y, L, do_plot=False)

        simp_x = simplification.x_pivots
        simp_y = simplification.y_pivots
        real_seg = simplification.num_real_segments
        if first:
            plt.figure()
            plt.title("Simplification using LSF")
            plt.plot(ts_x, ts_y, label="Original")
            plt.plot(simp_x, simp_y, label="Simplified")
            plt.plot(ts_x, SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)).line_version,
                     label="Line Simplified", linestyle="--")
            plt.show()
            first = False
        ts_simplifications.append(SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y), num_real_segments=real_seg))

    return ts_simplifications

def main():
    X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Y = [6, 3, 3, 5, 8, 6, 6, 7, 8, 9.1, 10]
    for i in range(1,len(X),1):
        L = i
        simp = LS.run(X, Y, L, do_plot=True, do_print=True)
        #simp = LS.run(X, Y, L)
        print(simp)
        print(simp.num_real_segments)

if __name__ == "__main__":
    main()