import numpy
import numpy as np
from numpy.random import PCG64  # Selecting training instances

from typing import List, Tuple
from tqdm import tqdm

from models.loadModel import model_classify, model_batch_classify

from localRobustness.LocalRobustNess import get_local_robust_score_approx

from dataSet.load_data import load_dataset_ts

from simplify.DPcustomAlgoKSmallest import solve_and_find_points

from utils.line import convert_all_points_to_lines, euclidean_distance_weighted
from utils.data import dataset_sensitive_c, set_c_manually
from utils.model import class_to_color
from utils.hard_coded_ts import get_hard_coded_ts, get_proto_types_index
from utils.folder import make_folder
from utils.scoring_functions import score_closeness, score_simplicity

from visualization.getTSParam import get_ts_param_org, get_ts_param_approx
from visualization.getEllipseParam import make_all_ellipse_param, both_in_and_out_ellipse_params
from visualization.plotting import PlotParams

from Perturbations.dataTypes import SegmentedTS

from ProtoTypes.ProtoTypes import get_prototypes


def get_simplicity_score(approximation, c):
    return score_simplicity(approximation, c)


def get_closeness_score(ts1: List[float] | np.ndarray, ts2: List[float] | np.ndarray, distance_weight: float,
                        alpha: float):
    # This should use the same function as the DP algo!!
    similarity = score_closeness(ts1=ts1, ts2=ts2, distance_weight=distance_weight, alpha=alpha)
    return similarity


IDX = 0


def score_approximation(approximation: SegmentedTS, original: List[float] | np.ndarray, model_name, target_class: int,
                        best_so_far_score: float, c: float,
                        distance_weight: float, alpha: float, gamma: float, resolution: int, robustness_title: str = "",
                        lim_y=None,
                        lim_x=None, original_ts: List[float] = None):
    """
    Score an input approximation
    :param alpha:
    :param distance_weight:
    :param best_so_far_score:
    :param c:
    :param target_class:
    :param approximation:
    :param original:
    :param model_name:
    :return score:
    """
    global IDX
    # resolution = 10 ** 3
    closeness = get_closeness_score(ts1=approximation.line_version, ts2=original, distance_weight=distance_weight,
                                    alpha=alpha)
    simplicity = get_simplicity_score(approximation, c)

    if closeness + simplicity > best_so_far_score:
        print(f"{gamma} gamma")
        if gamma < 0:
            if closeness * alpha + simplicity + gamma > best_so_far_score:
                return float("inf"), None, None, None

        else:
            # We can never do better so search should end.
            print("Closeness and simplicity score to large")
            return float("inf"), None, None, None
    IDX += 1

    epsilon = distance_weight * 0.1  # We change the current TS by 0.1 * max diff in dataset at each point
    robustness_score = get_local_robust_score_approx(approximation, model_name, k=resolution, target_class=target_class,
                                                     epsilon=epsilon, verbose=False, title=robustness_title,
                                                     save_file=str(IDX),
                                                     lim_x=lim_x, lim_y=lim_y,
                                                     original_ts=original_ts)
    robustness_score = 1 - robustness_score  # Robustness of 1 means every single perturbation had the target class
    score = closeness + simplicity + gamma * robustness_score
    return score, closeness, simplicity, robustness_score


def get_best_approximation_for_ts(single_ts: List[float] | np.ndarray, dataset_name: str, model_name: str,
                                  min_y: float, max_y: float,
                                  distance_weight: float, save_file_name: str, alpha: float, k: int,
                                  robustness_resolution: int,
                                  verbose: bool = False,
                                  gamma=1,
                                  early_stop: bool = True
                                  ) -> SegmentedTS:
    ts_length = len(single_ts)
    target_class = model_classify(model_name=model_name, time_series=single_ts)
    c = dataset_sensitive_c(dataset=dataset_name, distance_weight=distance_weight)
    nr_of_approximation = k

    print("what is going on?", alpha, c, nr_of_approximation, distance_weight)
    all_selected_points, all_ys = solve_and_find_points(
        X=list(range(len(single_ts))),
        Y=single_ts, c=c,
        K=nr_of_approximation,
        distance_weight=distance_weight,
        alpha=alpha
    )
    ts_segments = [SegmentedTS(x_pivots=x_pivots, y_pivots=y_pivots, ts_length=ts_length) for x_pivots, y_pivots in
                   zip(all_selected_points, all_ys)]
    all_lines = [segTS.line_version for segTS in ts_segments]
    all_pred_class = model_batch_classify(model_name=model_name, batch_of_timeseries=all_lines)
    for segTS, pred_class in zip(ts_segments, all_pred_class):  # type: SegmentedTS,int
        segTS.set_class(pred_class)

    print(f"Pre filter size: {len(ts_segments)}")
    # Filter away wrong class
    ts_segments = [segTS for segTS in ts_segments if segTS.pred_class == target_class]
    print(f"Post filter size: {len(ts_segments)}")
    if len(ts_segments) == 0:
        return None
    import matplotlib.pyplot as plt

    best_approx = None
    best_approx_score = float("inf")
    best_closeness = float("inf")
    best_simplicity = float("inf")
    best_robustness_score = float("inf")
    if verbose:
        all_folder = f"PyPlots/{dataset_name}/all_segs"
        make_folder(all_folder)
    for nr, approximation in tqdm(enumerate(ts_segments)):
        score, closeness, simplicity, robustness_score = score_approximation(
            approximation=approximation,
            original=single_ts,
            model_name=model_name,
            target_class=target_class,
            best_so_far_score=best_approx_score,
            distance_weight=distance_weight,
            c=c,
            alpha=alpha,
            gamma=gamma,
            resolution=robustness_resolution,
            robustness_title=dataset_name,
            lim_y=(min_y, max_y),
            lim_x=(0, 24),
            original_ts=single_ts
        )

        if closeness is None or simplicity is None or robustness_score is None:
            print("We break now!")
            break
        # Should be correct, as this only happens when closeness is > best
        if score < best_approx_score:
            best_approx = approximation
            best_approx_score = score
            best_closeness = closeness
            best_simplicity = simplicity
            best_robustness_score = robustness_score
            print(
                f"Best_score {best_approx_score}, best_approx {best_approx}\nCloseness: {best_closeness} simplicity: {best_simplicity} robustness: {best_robustness_score}")
            if verbose:
                make_plot(
                    original_ts=single_ts,
                    model_name=model_name,
                    title="",
                    y_lim=(min_y, max_y),
                    x_lim=(-1, ts_length),
                    save_file_name=f"{dataset_name}/all_segs/{nr}_{str(approximation.x_pivots)}",
                    best_approx=approximation,
                    display=False
                )
                print(nr)
    make_plot(
        original_ts=single_ts,
        model_name=model_name,
        title="",
        y_lim=(min_y, max_y),
        x_lim=(-1, ts_length),
        save_file_name=save_file_name,
        best_approx=best_approx
    )
    return best_approx
    # ts_param_org = get_ts_param_org(y_org=single_ts, model_name=model_name)
    # ts_param_approx = get_ts_param_approx(y_approx=best_approx.line_version, model_name=model_name)


#
# all_ellipse = both_in_and_out_ellipse_params(approximation=best_approx)
# title = ""  # f"Score {best_approx_score:.2f}, Closeness: {best_closeness:.2f} simplicity: {best_simplicity:.2f} robustness: {best_robustness_score:.2f} c:{c:.2f}"
# PlotParams(
#    ts_params=[ts_param_org, ts_param_approx],
#    ellipse_params=all_ellipse,
#    title=title,
#    y_lim=
#    x_lim=
#    display=False,
#    save_file=f"{save_file_name}"
# ).make_plot()


def make_plot(original_ts: List[float] | np.ndarray, model_name: str, title: str, y_lim: Tuple[float, float],
              x_lim: Tuple[int, int],
              save_file_name: str,
              best_approx: SegmentedTS,
              display=False):
    ts_param_org = get_ts_param_org(y_org=original_ts, model_name=model_name)
    ts_param_approx = get_ts_param_approx(y_approx=best_approx.line_version, model_name=model_name)

    all_ellipse = both_in_and_out_ellipse_params(approximation=best_approx)
    # title = ""  # f"Score {best_approx_score:.2f}, Closeness: {best_closeness:.2f} simplicity: {best_simplicity:.2f} robustness: {best_robustness_score:.2f} c:{c:.2f}"
    PlotParams(
        ts_params=[ts_param_org, ts_param_approx],
        ellipse_params=all_ellipse,
        title=title,
        y_lim=y_lim,
        x_lim=x_lim,
        display=display,
        save_file=f"{save_file_name}"
    ).make_plot()


def test():
    datasets = ["ItalyPowerDemand", "Chinatown", "ECG200"]
    model_names = ["ItalyPowerDemand_100.keras", "Chinatown_100.keras", "ECG200_100.keras"]
    c_values = [0.01]  # [0.0001, 0.001, 0.01, 0.1, 1]
    alpha_factors = [0.5]  # [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    robustness_resolution = 10 ** 3
    nr_of_approximations = 10 ** 5

    verbose = False

    for alpha in alpha_factors:
        for c in c_values:
            set_c_manually(c)
            for dataset_name, model_name in zip(datasets, model_names):
                all_ts = load_dataset_ts(dataset_name, data_type="TEST")
                min_y = min([min([y for y in ts]) for ts in all_ts])
                max_y = max([max([y for y in ts]) for ts in all_ts])

                distance_weight = abs(max_y - min_y)

                # Proto types
                selected_idx_class_0, selected_idx_class_1 = get_prototypes(dataset_name=dataset_name,
                                                                            model_name=model_name)
                proto_types_selected = selected_idx_class_0 + selected_idx_class_1
                for index in zip(proto_types_selected):
                    ts_single = all_ts[index]
                    # proto_type = get_hard_coded_ts(dataset=dataset_name, index=proto_type_nr)
                    get_best_approximation_for_ts(
                        single_ts=ts_single, model_name=model_name,
                        distance_weight=distance_weight, min_y=min_y, max_y=max_y,
                        dataset_name=dataset_name,
                        save_file_name=f"{dataset_name}/Train/proto_{index}",
                        alpha=alpha,
                        k=nr_of_approximations,
                        robustness_resolution=robustness_resolution,
                        verbose=verbose
                    )


def get_best_approximation(dataset_name: str, model_name: str, instance_nr: int, alpha=0.5, beta=0.01, gamma=1,
                           early_stop=True, k=2 * 10 ** 4, verbose=False, robustness_resolution: float = 5 * 10 ** 3):
    if not early_stop:
        print("Not early stopping mode")
    c_value = beta
    set_c_manually(c_value)
    robustness_resolution = robustness_resolution
    nr_of_approximations = k

    all_ts = load_dataset_ts(dataset_name, data_type="TEST")
    min_y = min([min([y for y in ts]) for ts in all_ts])
    max_y = max([max([y for y in ts]) for ts in all_ts])

    distance_weight = abs(max_y - min_y)

    original_ts = all_ts[instance_nr]
    best_approximation = get_best_approximation_for_ts(
        single_ts=original_ts, model_name=model_name,
        distance_weight=distance_weight, min_y=min_y, max_y=max_y,
        dataset_name=dataset_name,
        save_file_name=f"{dataset_name}/Train/proto_{instance_nr}",
        alpha=alpha,
        gamma=gamma,
        k=nr_of_approximations,
        robustness_resolution=robustness_resolution,
        early_stop=early_stop,
        verbose=verbose
    )
    return best_approximation


if __name__ == '__main__':
    test()
    # import cProfile
    # import pstats

    # cProfile.run('test()', "my_func_stats")
    # p = pstats.Stats("my_func_stats")
    # p.sort_stats("cumulative").print_stats()
