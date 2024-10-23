import random
import numpy as np
from typing import List, Tuple
from collections import defaultdict

from dataSet.load_data import load_dataset_ts
from simplify.DPcustomAlgoKSmallest import solve_and_find_points
from utils.line import check_if_all_is_on_line, get_pivot_points
import matplotlib.pyplot as plt
from utils.data import dataset_sensitive_c, get_min_and_max
from utils.line import interpolate_points_to_line
from models.loadModel import model_batch_classify, model_classify

from Perturbations.dataTypes import SegmentedTS
from Perturbations.singePointPerturbation import create_x_y_perturbation as single_perturbation
from Perturbations.fullLinePerturbation import create_x_y_perturbations as full_perturbation

from utils.model import class_to_alpha_color

from visualization.plotting import TSParam, PlotParams
from visualization.getEllipseParam import both_in_and_out_ellipse_params


def float_to_rgb(value):
    if value < 0 or value > 1:
        raise ValueError("Input value must be a float between 0 and 1.")
    cmap = plt.get_cmap("RdBu")
    rgb = cmap(value)[:3]
    return tuple([int(x * 255) for x in rgb])


# def create_permutations(points_y, max_y, min_y, k=10 ** 6, e=0.08):
#     dataset_dist = abs(max_y - min_y)
#     epsilon = e
#
#     change_range = dataset_dist * epsilon
#     permutations = []
#     for i in range(k):
#         new_val_points = []
#         for point in points_y:
#             random_val = random.gauss(0, 1)  # Currently using gaussion random, could use uniform random.uniform(-1, 1))
#             new_val_points.append(point + change_range * random_val)
#         permutations.append(new_val_points)
#
#     return permutations


def get_local_robust_score_approx(approximation: SegmentedTS, model_name: str, k: int, target_class: int,
                                  epsilon: float = None, verbose: bool = False, title: str = "",
                                  save_file: str = "", lim_y: Tuple[float, float] = None,
                                  lim_x: Tuple[float, float] = None, original_ts: List[float] = None) -> float:
    return get_local_robust_score(approximation=approximation,
                                  model_name=model_name, k=k, target_class=target_class, epsilon=epsilon,
                                  verbose=verbose, title=title, save_file=save_file, lim_y=lim_y, lim_x=lim_x,
                                  original_ts=original_ts)


def get_local_robust_score(approximation: SegmentedTS, model_name: str, target_class: int, k=10 ** 4,
                           random_seed=42,
                           verbose=False, title: str = "", save_file: str = "", lim_x=None, lim_y=None,
                           epsilon: float = None, original_ts: List[float] = None) -> float:
    """
    TODO: THIS SHOULD NOT DO ANY FORM OF PLOTTING INSIDE THE ROBUSTNESS SCORE!
    Returns the local robustness score of given approximation defined by its points of change.
    :param save_file:
    :param epsilon:
    :param approximation:
    :param target_class:
    :param lim_y:
    :param lim_x:
    :param title:
    :param verbose:
    :param random_seed:
    :param model_name:
    :param k:
    :return:
    """
    random.seed(random_seed)
    pivots_x = approximation.x_pivots
    ts_length = len(approximation.line_version)
    pivots_x = [0] + pivots_x[1:-1] + [ts_length - 1]  # We want to change the first and last point in the TS

    pivots_y = np.concatenate(
        [[approximation.line_version[0]], approximation.y_pivots[1:-1],
         [approximation.line_version[-1]]])  # approximation.y_pivots
    min_y = min(pivots_y)
    max_y = max(pivots_y)
    if epsilon is None:
        epsilon = (max_y - min_y) * 0.1

    # Single point?
    # all_permutation_points = single_perturbation(org_pivots_y=pivots_y, org_pivots_x=pivots_x, ts_length=ts_length,
    #                                             epsilon=epsilon)
    # Or all points
    all_permutation_points = full_perturbation(org_pivots_y=pivots_y, org_pivots_x=pivots_x, ts_length=ts_length,
                                               epsilon=epsilon, k=k)

    # Get class of all of them
    class_of_all = np.array(model_batch_classify(model_name=model_name,
                                                 batch_of_timeseries=[permutation.line_version for permutation in
                                                                      all_permutation_points]))
    unique, counts = np.unique(class_of_all, return_counts=True)
    dict_count = defaultdict(lambda: 0)
    for u, count in zip(unique, counts):
        if verbose:
            print(f"Number {u} occurs {count} times")
        dict_count[u] = count
    if verbose:
        print("Saving Fragility map")
        plt.clf()
        all_ts_plots = []
        for class_of_line, line in zip(class_of_all, all_permutation_points):  # type: int, SegmentedTS
            alpha = 0.05
            color = class_to_alpha_color(class_of_line, alpha=alpha)
            new_ts_parm = TSParam(
                x_values=list(range(ts_length)),
                y_values=line.line_version,
                color=color
            )
            all_ts_plots.append(new_ts_parm)
        # Approximation
        all_ts_plots.append(TSParam(
            x_values=list(range(ts_length)),
            y_values=approximation.line_version,
            color="black",  # class_to_alpha_color(predicted_class=target_class, alpha=1),
            linestyle=(2, (5, 2)),
            linewidth=4
        ))
        # Original
        if original_ts is not None:
            all_ts_plots.append(TSParam(
                x_values=list(range(ts_length)),
                y_values=original_ts,
                color="grey",
                linewidth=2

            ))

        all_ellipse = both_in_and_out_ellipse_params(approximation=approximation)
        PlotParams(
            ts_params=all_ts_plots,
            ellipse_params=all_ellipse,
            x_lim=lim_x,
            y_lim=lim_y,
            title="",
            # TODO: REDO title + " " + f"Fragility: {(1 - dict_count[target_class] / (dict_count[0] + dict_count[1])):.2f}",
            save_file=f"{save_file}"
        ).make_plot()
        # plt.title(title)
        # plt.xlim(lim_x)
        # plt.ylim(lim_y)
        # plt.savefig(f"pdfs/{random_seed}")

        # plt.show()

    return dict_count[target_class] / (dict_count[0] + dict_count[1])


def run():
    model_name = "Chinatown_1000.keras"
    dataset_name = "Chinatown"
    all_time_series = load_dataset_ts(dataset_name, data_type="TEST")
    min_y, max_y = get_min_and_max(all_time_series)
    plot_min_y = min_y - (max_y - min_y) / 4
    plot_max_y = max_y + (max_y - min_y) / 4
    plot_min_x = -1
    plot_max_x = len(all_time_series[0]) - 1

    lim_y = (plot_min_y, plot_max_y)
    lim_x = (plot_min_x, plot_max_x)

    instance_nr = 0
    ts = all_time_series[instance_nr]
    x_values = list(range(len(ts)))
    c_percentage = 200
    my_k = 50
    my_c = dataset_sensitive_c(all_time_series, c_percentage)  # Chinatown: 1
    min_robustness_score = float("inf")
    min_robustness_points = None
    min_idx = -1

    max_robustness_score = float("-inf")
    max_robustness_points = None
    max_idx = -1

    classes = model_batch_classify(model_name=model_name, batch_of_timeseries=[ts])[0]

    rob_loc_k = 10 ** 4
    random.seed(42)

    all_selected_points, all_ys = solve_and_find_points(X=x_values, Y=ts, c=my_c, K=my_k, saveImg=False)
    for i, selected_x in enumerate(all_selected_points):
        print(f"{i}/{len(all_selected_points)}")
        if selected_x != sorted(selected_x):
            print("What?")
        points_y = [ts[x] for x in selected_x]  # Extract y values we are going to change
        robustness_score = get_local_robust_score(ts_length=len(ts), in_points_x=selected_x, points_y=points_y,
                                                  model_name=model_name, k=rob_loc_k, random_seed=i,
                                                  target_class=classes)
        if robustness_score > max_robustness_score:
            max_robustness_score = robustness_score
            max_robustness_points = selected_x
            max_idx = i
        if robustness_score < min_robustness_score:
            min_robustness_score = robustness_score
            min_robustness_points = selected_x
            min_idx = i
    min_points_line = interpolate_points_to_line(ts_length=len(ts), x_selected=min_robustness_points,
                                                 y_selected=[ts[i] for i in min_robustness_points])
    max_points_line = interpolate_points_to_line(ts_length=len(ts), x_selected=max_robustness_points,
                                                 y_selected=[ts[i] for i in max_robustness_points])

    # Plot and verify that the scores are correct
    min_points_y = [ts[x] for x in min_robustness_points]  # Extract y values
    curr_class = model_classify(model_name=model_name, time_series=min_points_line)
    robustness_score_min = get_local_robust_score(ts_length=len(ts), in_points_x=min_robustness_points,
                                                  points_y=min_points_y,
                                                  model_name=model_name, k=rob_loc_k, random_seed=min_idx,
                                                  verbose=True,
                                                  title=f"Local robustness of best approximation class 0 score: {1 - min_robustness_score:.2f}",
                                                  lim_x=lim_x,
                                                  lim_y=lim_y, target_class=curr_class)  # PLOT AND VERIFY

    max_points_y = [ts[x] for x in max_robustness_points]  # Extract y values
    curr_class = model_classify(model_name=model_name, time_series=max_points_line)
    robustness_score_max = get_local_robust_score(ts_length=len(ts), in_points_x=max_robustness_points,
                                                  points_y=max_points_y,
                                                  model_name=model_name, k=rob_loc_k, random_seed=max_idx,
                                                  verbose=True,
                                                  title=f"Local robustness of best approximation class 1 score: {max_robustness_score:.2f}",
                                                  lim_x=lim_x,
                                                  lim_y=lim_y,
                                                  target_class=curr_class
                                                  )  # PLOT AND VERIFY

    plt.clf()
    plt.plot(x_values, ts, color="black")
    plt.plot(x_values, min_points_line, "--b")
    plt.plot(x_values, max_points_line, "--r")
    plt.legend(["Original", "Max Robustness class 0", "Max Robustness class 1"])
    plt.xlim(lim_x)
    plt.ylim(lim_y)
    plt.title("Org TS, Max Robu. approx. class 1, Max Robu. approx. class 0")
    plt.savefig(f"pdfs/{instance_nr}_overview")

    plt.show()

    print(f"Min local robustness score: {min_robustness_score},  {robustness_score_min}")
    print(f"Max local robustness score: {max_robustness_score}, {robustness_score_max}")


if __name__ == '__main__':
    run()
