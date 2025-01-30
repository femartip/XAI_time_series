import numpy as np
from matplotlib import pyplot as plt
from rdp import rdp
import logging

from Utils.load_data import load_dataset
from ORSalgorithm.ORS_algorithm import get_simplifications
from ORSalgorithm.Perturbations.dataTypes import SegmentedTS
from SimplificationMethods.BottumUp.bottomUp import  get_swab_approx
from SimplificationMethods.Visvalingam_whyattt.Visvalingam_Whyatt import  simplify as VC_simplify


def get_OS_simplification(dataset_name, datset_type, alpha):
    """
    Apply OS algorithm to simplify all time series in the dataset.
    """
    all_time_series = load_dataset(dataset_name, data_type=datset_type)
    all_simplifications = []
    first = False
    logging.debug(alpha)
    for ts_y in all_time_series:
        ts_x = [i for i in range(len(ts_y))]
        my_k = 1
        beta = 1/(len(ts_x)-1)*(1-alpha)
        selcted_xs, selectd_ys = get_simplifications(X=ts_x, Y=ts_y, K=my_k, alpha=alpha, beta=beta)
       
        if first:
            plt.figure()
            plt.plot(ts_x, ts_y, label="Original")
            plt.plot(selcted_xs[0], selectd_ys[0], label="Simplified")
            plt.plot(ts_x, SegmentedTS(x_pivots=selcted_xs[0], y_pivots=selectd_ys[0], ts_length=len(ts_y)).line_version, label="Line Simplified", linestyle="--")
            plt.legend()
            plt.show()
       
            first = False
        segTS = SegmentedTS(x_pivots=selcted_xs[0], y_pivots=selectd_ys[0], ts_length=len(ts_y))
        all_simplifications.append(segTS)
    
    return all_time_series, all_simplifications


def get_RDP_simplification(dataset_name,datset_type, epsilon):
    """
    Apply Ramer-Douglas-Peucker (RDP) algorithm to simplify all time series in the dataset.
    """
    all_time_series = load_dataset(dataset_name, data_type=datset_type)
    all_simplifications = []
    first = False
    epsilon = epsilon
    logging.debug("epis:", epsilon)
    for ts_y in all_time_series:
        ts_x = list(range(len(ts_y)))
        simp_mask = rdp(np.array(list(zip(ts_x,ts_y))), epsilon=epsilon, return_mask=True)
        simp_y = [y for y,bool in zip(ts_y, simp_mask) if bool]
        simp_x = [x for x,bool in zip(ts_x, simp_mask) if bool]
        if first:
            plt.figure()
            plt.plot(ts_x, ts_y, label="Original")
            plt.plot(simp_x, simp_y, label="Simplified")
            plt.plot(ts_x, SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)).line_version, label="Line Simplified", linestyle="--")
            plt.show()
            first = False
        all_simplifications.append(SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)))
    
    return all_time_series, all_simplifications


def get_bottom_up_simplification(dataset_name, datset_type, max_error):
    """
    Apply Bottom Up algorithm to simplify all time series in the dataset.
    """
    all_time_series = load_dataset(dataset_name, data_type=datset_type)
    all_simplifications = []
    first = False
    max_error = max_error
    logging.debug("max_error:", max_error)
    for ts_y in all_time_series:
        ts_x = list(range(len(ts_y)))
        simplification = get_swab_approx(ts_y, max_error=max_error)
        simp_x = simplification.x_pivots
        simp_y = simplification.y_pivots
        if first:
            plt.figure()
            plt.title("Simplification using SWAB")
            plt.plot(ts_x, ts_y, label="Original")
            plt.plot(simp_x, simp_y, label="Simplified")
            plt.plot(ts_x, SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)).line_version,
                     label="Line Simplified", linestyle="--")
            plt.show()
            first = False
        all_simplifications.append(SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)))

    return all_time_series, all_simplifications

def get_VC_simplification(dataset_name, datset_type, alpha):
    """
    Apply Visvalingam Whyatt algorithm to simplify all time series in the dataset.
    """
    all_time_series = load_dataset(dataset_name, data_type=datset_type)
    all_simplifications = []
    first = True
    alpha = alpha
    logging.debug("alpha:", alpha)
    for ts_y in all_time_series:
        ts_x = list(range(len(ts_y)))
        simplification = VC_simplify(ts_y, alpha=alpha)
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
        all_simplifications.append(SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)))

    return all_time_series, all_simplifications