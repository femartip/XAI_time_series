from typing import List, Tuple
import random
from collections import defaultdict

from ORSalgorithm.simplify.DPcustomAlgoKSmallest import solve_and_find_points
from Utils.load_data import load_dataset
from ORSalgorithm.Perturbations.dataTypes import SegmentedTS


def get_min_and_max(ts_all) -> Tuple[float, float]:
    """
    returns min_y and max_y in dataset
    :param ts_all:
    :return: min_y, max_y
    """
    max_y = max([max(ts) for ts in ts_all])
    min_y = min([min(ts) for ts in ts_all])
    return min_y, max_y


def get_min_max_from_dataset_name(dataset_name: str) -> Tuple[float, float]:
    return get_min_and_max(load_dataset(dataset_name=dataset_name, data_type="TEST"))


def dataset_sensitive_c(distance_weight: float, dataset: str) -> float:
    c_percentage = get_c_percentage_by_dataset(dataset_name=dataset)
    c = distance_weight * c_percentage
    #print(f"c constant {c}")
    return c


def get_time_series(dataset_name: str, instance_nr: int):
    all_time_series = load_dataset(dataset_name, data_type="TEST")
    return all_time_series[instance_nr]


def get_time_series_and_seg_pivots(dataset_name, instance_nr=0):
    all_time_series = load_dataset(dataset_name, data_type="TEST")

    # instance_nr = 0
    ts = all_time_series[instance_nr]
    x_values = list(range(len(ts)))
    c_percentage = 200  # Chinatown : 200
    my_k = 1
    my_c = dataset_sensitive_c(all_time_series, c_percentage)  # Chinatown: 1

    random.seed(42)

    all_selected_points, all_ys = solve_and_find_points(X=x_values, Y=ts, c=my_c, K=my_k, saveImg=False)
    best_fit_points = all_selected_points[0]
    best_fit_ys = all_ys[0]

    min_y, max_y = get_min_and_max(all_time_series)
    min_y = min_y - (max_y - min_y) / 4  # Extra buffer
    max_y = max_y + (max_y - min_y) / 4  # Extra buffer

    return best_fit_points, best_fit_ys, ts, min_y, max_y


manually_datasets = {
    "Chinatown": 0.01,
    "ItalyPowerDemand": 0.01,
    "ECG200": 0.01
}


def set_c_manually(c_value, dataset: str = None):
    global manually_datasets
    if dataset is None:
        for key in manually_datasets.keys():
            manually_datasets[key] = c_value
    else:
        manually_datasets[dataset] = c_value


def get_c_percentage_by_dataset(dataset_name: str) -> float:
    global manually_datasets
    c_percentage_by_dataset = defaultdict(lambda: 1.0)
    for key, value in manually_datasets.items():
        c_percentage_by_dataset[key] = value

    return c_percentage_by_dataset[dataset_name]


def get_best_segmentations_approximations(dataset_name: str, instance_nr: int = 0, top_k=1) -> List[SegmentedTS]:
    all_time_series = load_dataset(dataset_name, data_type="TEST")

    # instance_nr = 0
    ts = all_time_series[instance_nr]
    x_values = list(range(len(ts)))
    c_percentage = get_c_percentage_by_dataset(dataset_name=dataset_name)

    my_k = top_k
    my_c = dataset_sensitive_c(all_time_series, c_percentage)  # Chinatown: 1

    random.seed(42)

    all_x_pivots, all_y_pivots = solve_and_find_points(X=x_values, Y=ts, c=my_c, K=my_k, saveImg=False)
    top_k_approximations = []
    for x_pivots, y_pivots in zip(all_x_pivots, all_y_pivots):
        segmented_ts = SegmentedTS(x_pivots=x_pivots, y_pivots=y_pivots)
        segmented_ts.set_line_version(ts_length=len(ts))
        top_k_approximations.append(segmented_ts)

    return top_k_approximations
