import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import os

import numpy as np
import random
from typing import List, Dict, Tuple
from collections import defaultdict

from models.loadModel import model_batch_classify, model_classify
from utils.data import dataset_sensitive_c, get_min_and_max, get_time_series_and_seg_pivots
from utils.line import interpolate_points_to_line, convert_all_points_to_lines
from dataSet.load_data import load_dataset_ts

from utils.model import class_to_color
from visualization.plotting import ScatterParams, PlotParams
from visualization.getTSParam import get_ts_param_org, get_ts_param_approx
from visualization.getEllipseParam import make_all_ellipse_param

from Perturbations.dataTypes import SinglePointPerturbation


def create_x_y_perturbation(org_pivots_y: List[float], org_pivots_x: List[int], ts_length: int, epsilon: float) -> \
        List[SinglePointPerturbation]:
    resolution = 10 ** 3
    all_ys_perturbations = list(np.linspace(-epsilon, epsilon, resolution, endpoint=True))
    x_change = 1
    num_changes = x_change * 2 + 1
    all_x_perturbations = list(np.linspace(-x_change, x_change, num_changes, endpoint=True))

    all_perturbations = []
    for idx in range(len(org_pivots_x)):
        for x_change in all_x_perturbations:
            if x_change != 0 and (x_change + org_pivots_x[idx] in org_pivots_x or x_change + org_pivots_x[idx] < 0
                                  or x_change + org_pivots_x[idx] >= ts_length):
                continue
                # We don't want to change x so much it overlaps with another x in org_pivots_x
                # NB: We can still have that two perturbations overlap
                # NB: Don't change x out of the range [0 ts_length-1]
            for y_change in all_ys_perturbations:
                new_y_value = org_pivots_y[idx] + y_change
                new_x_value = org_pivots_x[idx] + x_change
                new_y_pivots = org_pivots_y[:idx] + [new_y_value] + org_pivots_y[idx + 1:]
                new_x_pivots = org_pivots_x[:idx] + [new_x_value] + org_pivots_x[idx + 1:]
                new_perturbation = SinglePointPerturbation(new_x=new_x_value, new_y=new_y_value, idx_pivots=idx,
                                                           x_pivots=new_x_pivots, y_pivots=new_y_pivots,
                                                           ts_length=ts_length)
                all_perturbations.append(new_perturbation)

    return all_perturbations


def make_perturbations(pivots_y_original: List[float], pivots_x_original: List[int],
                       line_version_original: List[float], epsilon: float) -> List[SinglePointPerturbation]:
    """
    We define a perturbation to be a combination of two changes.
    1. Changing a y value in pivots_y. I.e. pivots_y = [0,0,2,2,1] -> [0,0,4,2,2,1]
    2. Changing an x value in pivots_x. I.e. pivots_x = [0,2,5,8,10] -> [0,2,3,8,10]

    To identify a perturbation we will need a tuple of 3 items:
    Index of pivot to change, x changed of pivot, y changed of pivot.

    example:
    Og.piv_x = [0,5,10], Og.piv_y = [10,5,10]
    perturbation: change middle higher and more right->
    Per.piv_x = [0,7,10], Per.piv_y = [10,7,10]
    For each perturbation we create a SinglePointPerturbation.
    This store piv_x, piv_y, idx, line_version,pred_class.


    Finally, we will return a list of all the SinglePointPerturbation.
    :return: List[SinglePointPerturbation].
    """
    org_ts_length = len(line_version_original)
    all_perturbations = create_x_y_perturbation(org_pivots_y=pivots_y_original, org_pivots_x=pivots_x_original,
                                                ts_length=org_ts_length, epsilon=epsilon)
    # calculate line version
    for perturbation in all_perturbations:
        perturbation.set_line_version(ts_length=org_ts_length)
    return all_perturbations


def make_perturbations_and_get_class(pivots_y_original: List[float], pivots_x_original: List[int],
                                     line_version_original: List[float] | np.ndarray, epsilon: float, model_name: str):
    # Make perturbations
    all_perturbations = make_perturbations(pivots_x_original=pivots_x_original, pivots_y_original=pivots_y_original,
                                           line_version_original=line_version_original, epsilon=epsilon)
    # get class
    all_lines = [perturbation.perturbationTS.line_version for perturbation in all_perturbations]

    all_pred_class = model_batch_classify(model_name=model_name, batch_of_timeseries=all_lines)
    for perturbation, pred_class in zip(all_perturbations, all_pred_class):
        perturbation.set_class(pred_class)

    return all_perturbations


def get_perturbations_scatter_params(all_perturbations: List[SinglePointPerturbation]) -> List[ScatterParams]:
    all_scatter_params = {
        0: [],
        1: []
    }
    for perturbation in all_perturbations:
        if perturbation.perturbationTS.pred_class == 0:
            all_scatter_params[0].append(perturbation)
        else:
            all_scatter_params[1].append(perturbation)

    all_0_x = [perturbation.new_x for perturbation in all_scatter_params[0]]
    all_0_y = [perturbation.new_y for perturbation in all_scatter_params[0]]

    all_1_x = [perturbation.new_x for perturbation in all_scatter_params[1]]
    all_1_y = [perturbation.new_y for perturbation in all_scatter_params[1]]

    alpha = 2e-3
    scatter_0 = ScatterParams(x_values=all_0_x, y_values=all_0_y,
                              color=to_rgba(class_to_color(0), alpha=alpha))
    scatter_1 = ScatterParams(x_values=all_1_x, y_values=all_1_y, color=to_rgba(class_to_color(1), alpha=alpha))

    scatter_params = [scatter_0, scatter_1]

    return scatter_params


def plot_perturbations_org_approx(perturbations: List[SinglePointPerturbation], original_ts: List[float] | np.ndarray,
                                  approximation_ts: List[float], model_name: str, pivot_x_org: List[int],
                                  pivot_y_org: List[float],
                                  pivot_x_approx: List[int], pivot_y_approx: List[int]):
    scatter_params = get_perturbations_scatter_params(perturbations)

    # Get TS param for original
    ts_param_org = get_ts_param_org(y_org=original_ts, model_name=model_name)
    # Get TS Param for approx
    ts_param_approx = get_ts_param_approx(y_approx=approximation_ts, model_name=model_name)
    ts_params = [ts_param_org, ts_param_approx]

    # Add ellipses on the pivot points
    ellipsis_org = make_all_ellipse_param(x_pivots=pivot_x_org, y_pivots=pivot_y_org, inner=True)
    # Add hallow ellipses on the edge points
    ellipsis_approx = make_all_ellipse_param(x_pivots=pivot_x_approx, y_pivots=pivot_y_approx, inner=False)
    all_ellipse_params = ellipsis_org + ellipsis_approx

    # Make the plot and display it
    title = "Best Title there is!"
    save_file = "testFile"
    x_min = min(perturbation.new_x for perturbation in perturbations)
    x_max = max(perturbation.new_x for perturbation in perturbations)

    y_min = min(perturbation.new_y for perturbation in perturbations)
    y_max = max(perturbation.new_y for perturbation in perturbations)

    x_lim = (x_min - 1, x_max + 1)
    y_lim = (y_min - (y_max - y_min) / 10, y_max + (y_max - y_min) / 10)  # Some extra room

    PlotParams(ts_params=ts_params, scatter_params=scatter_params, ellipse_params=all_ellipse_params, title=title,
               save_file=save_file, display=True,
               x_lim=x_lim,
               y_lim=y_lim).make_plot()


def test(instance_nr, model_name, dataset_name):
    pivot_points_x, pivot_points_y, ts, min_y, max_y = get_time_series_and_seg_pivots(dataset_name=dataset_name,
                                                                                      instance_nr=instance_nr)
    pivot_point_x_org = pivot_points_x.copy()
    pivot_point_y_org = pivot_points_y.copy()
    # best_fit_ys_org = pivot_points_y.copy()
    # best_fit_points_org = pivot_point_x.copy()
    approx_line = interpolate_points_to_line(len(ts), pivot_points_x, pivot_points_y)
    pivot_points_x[0] = 0
    pivot_points_x[-1] = len(ts) - 1

    pivot_points_y[0] = approx_line[0]
    pivot_points_y[-1] = approx_line[-1]

    epsilon_div = (max_y - min_y) / 5
    all_perturbations = make_perturbations_and_get_class(pivots_x_original=pivot_points_x,
                                                         pivots_y_original=pivot_points_y,
                                                         line_version_original=ts, model_name=model_name,
                                                         epsilon=epsilon_div)
    plot_perturbations_org_approx(perturbations=all_perturbations, original_ts=ts, approximation_ts=approx_line,
                                  model_name=model_name, pivot_x_org=pivot_point_x_org, pivot_y_org=pivot_point_y_org,
                                  pivot_x_approx=pivot_points_x, pivot_y_approx=pivot_points_y)


def run():
    model_name = "Chinatown_1000.keras"
    dataset_name = "Chinatown"
    os.makedirs(f"PyPlots/{dataset_name}", exist_ok=True)
    for i in range(0, 100, 1):
        test(instance_nr=i, model_name=model_name, dataset_name=dataset_name)


if __name__ == '__main__':
    run()
