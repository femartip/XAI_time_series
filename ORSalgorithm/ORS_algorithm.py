import logging
import numpy as np
import random

import ORSalgorithm.simplify.DPcustomAlgoKSmallest
from ORSalgorithm.simplify.DPcustomAlgoKSmallest import solve_and_find_points
from ORSalgorithm.Utils.data import get_min_and_max, dataset_sensitive_c
from ORSalgorithm.Utils.line import interpolate_points_to_line
from ORSalgorithm.Utils.loadModel import model_classify, model_confidence, model_batch_classify, batch_confidence

logging.basicConfig(level=logging.INFO)

def ORSalgorithm(time_series, model_path, k=10000, alpha=0.02):             #𝑜𝑝𝑡_𝑠𝑖𝑚𝑝(𝑡𝑠(time_series), ℎ(model), 𝛼, 𝛽, 𝛾)
    """
    time_series: list of time series as numpy arrays
    model_path: path to the model to be used for classification
    k: number of best solutions to keep, do not confuse with the k defined in the paper
    alpha: Adjusts how strictly the algorithm penalizes deviations from the line fit. Parameter for the ORS algorithm, incluences the score closeness = error * alpha / distance_weight

    Returns: list of best simplified and robust approximations for each time series
    """
    logging.debug(f"Shape of time series:{time_series.shape}")
    min_y, max_y = get_min_and_max(time_series)   # Get the minimum and maximum y values
    distance_weight = max_y - min_y     # Distance of the y values
    c = distance_weight * 0.01      # Showes in the paper as 𝜖, calculated by 10% of the y-range, this gives flexibility for setting the simplified initial and final points
    
    chosen_simplifications = []
    confidence_chosen_simplifications = []
    class_chosen_simplifications = []

    for ts_nr, ts in enumerate(time_series):
        """
        STAGE 1
        DP Scheme for simplifications for all time series in the dataset
        This is model independant of the model 
        Use DP to compute a 2-dimensional table of size n x q that in cell D[j,k] stores the kth least costly simplification for the points p1, ..., pj of the input time series ts, 
        considering only error and simplicity but not robustness, as in above DP.
        """
        logging.debug(f"TS number: {ts_nr}")
        logging.debug(f"TS: {ts}")

        x_values = [i for i in range(len(ts))]

        all_selected_points, all_ys = solve_and_find_points(x_values, ts, c=c, K=k, saveImg=False, distance_weight=distance_weight, alpha=alpha)    # Dim (k, 3) -> 3 for tuple value, index, rank
        all_interpolations = []

        # Constructs table of size n x q        
        for i, (selected_points, ys) in enumerate(zip(all_selected_points, all_ys)):
            inter_ts = interpolate_points_to_line(ts_length=len(ts), x_selected=selected_points, y_selected=ys)
            all_interpolations.append(inter_ts)     # Dim (k, len(ts))

        """
        STAGE 2
        Robustness check
        This is dependant on the model
        Consider the q least costly simplifications sts1, ..., stsq, ordered by non-decreasing cost under error and simplicity as stored in D[n, 1]..D[n, q] respectively. 
        Compute robustness for any simplification that h classifies same as ts.
        """ 
        org_class = model_classify(model_path, ts)
        org_confidence = model_confidence(model_path, ts)
        all_classes = model_batch_classify(model_path, all_interpolations)

        ts_and_class = zip(all_classes, list(range(len(all_interpolations))))  # Select segmentations with same classification

        ts_idx_to_keep = list(map(lambda x: x[1], filter(lambda x: x[0] == org_class, ts_and_class)))  # Select the indices of the time series that have the same class as the original time series

        confidence_of_keep = batch_confidence(model_path=model_path, batch_of_timeseries=list(map(lambda x: all_interpolations[x], ts_idx_to_keep)))    # Compute confidence of the selected time series

        highest_confidence_among_keep_idx = np.argmax(confidence_of_keep)  # Find the highest confidence among the selected time series
        highest_confidence_idx = ts_idx_to_keep[highest_confidence_among_keep_idx]  # Extract the idx

        class_approx = model_classify(model_path, all_interpolations[highest_confidence_idx])   # Compute the class of the selected time series, AGAIN?
        confidence_approx = confidence_of_keep[highest_confidence_among_keep_idx]   # Compute the confidence of the selected time series
        
        logging.debug(f"Original class: {org_class}, Original confidence: {org_confidence}")
        logging.debug(f"TS idx to keep: {ts_idx_to_keep}, Approx class: {class_approx}, Approx confidence: {confidence_approx}")

        chosen_simplifications.append(all_interpolations[highest_confidence_idx])
        confidence_chosen_simplifications.append(confidence_approx)
        class_chosen_simplifications.append(class_approx)
    
    logging.debug(f"Average confidence of all approximations: {np.mean(confidence_chosen_simplifications)}")
    return chosen_simplifications, confidence_chosen_simplifications, class_chosen_simplifications
