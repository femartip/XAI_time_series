import random
from typing import List
import numpy as np

from matplotlib.colors import to_rgba

from Perturbations.dataTypes import SegmentedTS
from models.loadModel import model_batch_classify

from utils.model import class_to_color
from utils.data import get_time_series_and_seg_pivots
from utils.line import interpolate_points_to_line

from visualization.plotting import TSParam, PlotParams


def create_x_y_perturbations(org_pivots_y: List[float], org_pivots_x: List[int], ts_length: int, epsilon: float,
                             k: int = 5e4) -> \
        List[SegmentedTS]:
    all_perturbations = []
    resolution = int(k)  # Number of lines
    random.seed(42)
    for i in range(resolution):
        new_pivots_y = []
        new_pivots_x = []
        for j in range(len(org_pivots_x)):
            # random_y_change = random.gauss(0, epsilon) Gaussion
            random_y_change = random.uniform(-epsilon, epsilon)
            new_pivot_y = org_pivots_y[j] + random_y_change

            x_range = 0
            possible_x_values = list(range(-x_range, x_range + 1))
            if j == 0:
                possible_x_values = list(range(0, x_range + 1))
            if j == len(org_pivots_x) - 1:
                possible_x_values = list(range(-x_range, 1))

            random.shuffle(possible_x_values)
            found_new_pivot_x = None
            for random_x_change in possible_x_values:
                new_pivot_x = org_pivots_x[j] + random_x_change
                if new_pivot_x in new_pivots_x or new_pivot_y in org_pivots_x:
                    continue
                else:
                    found_new_pivot_x = new_pivot_x
                    break

            new_pivots_y.append(new_pivot_y)
            new_pivots_x.append(found_new_pivot_x)
        tsParam = SegmentedTS(x_pivots=new_pivots_x, y_pivots=new_pivots_y, ts_length=ts_length)
        all_perturbations.append(tsParam)
    return all_perturbations


def classify_all_perturbations(all_perturbations: List[SegmentedTS], model_name: str) -> List[SegmentedTS]:
    all_line_version = [perturbation.line_version for perturbation in all_perturbations]
    all_pred_classes = model_batch_classify(batch_of_timeseries=all_line_version, model_name=model_name)
    for perturbation, pred_class in zip(all_perturbations, all_pred_classes):  # type: SegmentedTS, int
        perturbation.set_class(pred_class)

    return all_perturbations


def full_perturbations_to_ts_params(all_perturbations: List[SegmentedTS]) -> List[TSParam]:
    all_perturbation_params = []
    for perturbation in all_perturbations:
        alpha = 2e-3
        ts_param = TSParam(
            x_values=list(range(0, len(perturbation.line_version))),
            y_values=perturbation.line_version,
            color=to_rgba(
                class_to_color(perturbation.pred_class),
                alpha=alpha
            )
        )
        all_perturbation_params.append(ts_param)
    return all_perturbation_params


def plot_full_line_perturbations(all_perturbations: List[SegmentedTS], original_ts: List[float] | np.ndarray,
                                 approximation_ts: List[float], model_name: str, pivot_x_org: List[int],
                                 pivot_y_org: List[float],
                                 pivot_x_approx: List[int], pivot_y_approx: List[int]):
    all_perturbation_params = full_perturbations_to_ts_params(all_perturbations)
    PlotParams(ts_params=all_perturbation_params, display=True).make_plot()


def run(dataset_name, model_name, instance_nr):
    pivot_points_x, pivot_points_y, ts, min_y, max_y = get_time_series_and_seg_pivots(dataset_name=dataset_name,
                                                                                      instance_nr=instance_nr)
    approximation_line = interpolate_points_to_line(len(ts), pivot_points_x, pivot_points_y)
    epsilon = (max_y - min_y) / 10
    perturbations = create_x_y_perturbations(org_pivots_y=pivot_points_y, org_pivots_x=pivot_points_x,
                                             ts_length=len(ts), epsilon=epsilon)
    perturbations = classify_all_perturbations(all_perturbations=perturbations, model_name=model_name)
    plot_full_line_perturbations(all_perturbations=perturbations, original_ts=ts, approximation_ts=approximation_line,
                                 model_name=model_name, pivot_x_org=pivot_points_x, pivot_y_org=pivot_points_y,
                                 pivot_x_approx=pivot_points_x, pivot_y_approx=pivot_points_y)


def start_run():
    model_name = "Chinatown_1000.keras"
    dataset_name = "Chinatown"
    for i in range(0, 100, 1):
        run(instance_nr=i, model_name=model_name, dataset_name=dataset_name)


if __name__ == '__main__':
    start_run()
