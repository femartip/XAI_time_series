import logging
import numpy as np

from ORSalgorithm.simplify.DPcustomAlgoKSmallest import solve_and_find_points
from ORSalgorithm.Utils.data import get_min_and_max, dataset_sensitive_c
from ORSalgorithm.Utils.line import interpolate_points_to_line
from ORSalgorithm.Utils.loadModel import model_classify, model_confidence, model_batch_classify, batch_confidence

logging.basicConfig(level=logging.DEBUG)

def ORSalgorithm(time_series, model_path, k=10000, alpha=0.02):
    """
    time_series: list of time series as numpy arrays
    model_path: path to the model to be used for classification
    k: number of best solutions to keep
    alpha: parameter for the ORS algorithm

    Returns: list of best simplified and robust approximations for each time series
    """
    min_y, max_y = get_min_and_max(time_series)
    distance_weight = max_y - min_y

    #c = dataset_sensitive_c(dataset=dataset_name, distance_weight=distance_weight) 
    c = distance_weight * 0.01      #In practice both are equivalent, but do not know where the 0.01 comes from
    for ts_nr, ts in enumerate(time_series):
        """
        DP Scheme for simplifications for all time series in the dataset
        This is model independant of the model 
        """
        logging.debug("TS number:", ts_nr)
        logging.debug(f"TS: {ts}")

        x_values = [i for i in range(len(ts))]

        all_selected_points, all_ys = solve_and_find_points(x_values, ts, c=c, K=k, saveImg=False, distance_weight=distance_weight, alpha=alpha)
        all_interpolations = []
        
        for i, (selected_points, ys) in enumerate(zip(all_selected_points, all_ys)):
            inter_ts = interpolate_points_to_line(ts_length=len(ts), x_selected=selected_points, y_selected=ys)
            all_interpolations.append(inter_ts)

        """
        Robustness check
        This is dependant on the model
        """ 
        org_class = model_classify(model_path, ts)
        org_confidence = model_confidence(model_path, ts)
        all_classes = model_batch_classify(model_path, all_interpolations)
        all_confidence = batch_confidence(model_path, all_interpolations)

        # Select segmentations with same classification
        ts_and_class = zip(all_classes, list(range(len(all_interpolations))))

        ts_idx_to_keep = list(map(lambda x: x[1], filter(lambda x: x[0] == org_class, ts_and_class)))
        confidence_of_keep = batch_confidence(model_path=model_path, batch_of_timeseries=list(map(lambda x: all_interpolations[x], ts_idx_to_keep)))

        highest_confidence_among_keep_idx = np.argmax(confidence_of_keep)  # np.argmax(confidence_of_keep)
        highest_confidence_idx = ts_idx_to_keep[highest_confidence_among_keep_idx]  # Extract the idx

        class_approx = model_classify(model_path, all_interpolations[highest_confidence_idx])
        confidence_approx = confidence_of_keep[highest_confidence_among_keep_idx]
        
        logging.debug(f"Original class: {org_class}, Original confidence: {org_confidence}")
        logging.debug(f"Approx class: {class_approx}, Approx confidence: {confidence_approx}")
        logging.debug(f"TS idx to keep: {ts_idx_to_keep}")
        logging.debug(f"Confidence of keep: {confidence_of_keep}")

        return all_interpolations[highest_confidence_idx]