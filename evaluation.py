import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import datetime

from simplifications import get_OS_simplification, get_RDP_simplification, get_bottom_up_simplification, \
    get_VC_simplification, get_LSF_simplification
from Utils.metrics import calculate_mean_loyalty, calculate_kappa_loyalty, calculate_complexity, score_simplicity
from Utils.load_models import model_batch_classify # type: ignore
from Utils.load_data import load_dataset

logging.basicConfig(level=
logging.debug)


def score_different_alphas(dataset_name, datset_type, model_path):
    """
    Evaluate the impact of different alpha values on loyalty, kappa, and complexity.
    """
    diff_alpha_values = np.arange(0,1,0.01)
    df = pd.DataFrame(columns=["Type","Alpha", "Mean Loyalty", "Kappa Loyalty", "Complexity", "Num Segments"])
    all_time_series = load_dataset(dataset_name, data_type=datset_type)

    # Algorithms grow in time complexity with the number of time series. If there are too many, we will only use the first 100
    if np.shape(all_time_series)[0] > 100:
        real_shape = np.shape(all_time_series)
        all_time_series = all_time_series[:100]
        logging.info(f"Number of instances {real_shape} > 100, so modified to: {np.shape(all_time_series)[0]}")

    time_os = []
    time_rdp = []
    time_vc = []
    time_bu = []

    for alpha in tqdm(diff_alpha_values):
        # Step 1 gen all simplified ts
        logging.debug(f"Alpha: {alpha}")   
        
        logging.debug("Running OS")
        init_time = datetime.datetime.now()
        all_time_series_OS, all_simplifications_OS = get_OS_simplification(time_series=all_time_series, alpha=alpha)
        time_os.append((datetime.datetime.now()-init_time).total_seconds())

        #Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_OS]
        predicted_classes_simplifications_OS = get_model_predictions(model_path, batch_simplified_ts)
        predicted_classes_original = get_model_predictions(model_path, all_time_series_OS)

        # Step 3 calculate loyalty and complexity
        mean_loyalty_OS = calculate_mean_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_OS)
        kappa_loyalty_OS = calculate_kappa_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_OS)
        complexity_OS = calculate_complexity(batch_simplified_ts=all_simplifications_OS)
        num_segments_OS = np.mean([(len(ts.x_pivots) - 1) for ts in all_simplifications_OS])
        row = ["OS", alpha, mean_loyalty_OS, kappa_loyalty_OS, complexity_OS, num_segments_OS]
        df.loc[len(df)] = row

        # Step 1 gen all simplified ts
        
        logging.debug("Running RDP")
        init_time = datetime.datetime.now()
        all_time_series_RDP, all_simplifications_RDP = get_RDP_simplification(time_series=all_time_series, epsilon=alpha)
        time_rdp.append((datetime.datetime.now()-init_time).total_seconds())
        
        #Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_RDP]
        predicted_classes_simplifications_RDP = get_model_predictions(model_path, batch_simplified_ts)
        predicted_classes_original = get_model_predictions(model_path, all_time_series_RDP)     #I will say this and all_time_series_OS are the same, but just in case

        # Step 3 calculate loyalty and complexity
        mean_loyalty_RDP = calculate_mean_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_RDP)
        kappa_loyalty_RDP = calculate_kappa_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_RDP)
        complexity_RDP = calculate_complexity(batch_simplified_ts=all_simplifications_RDP)
        num_segments_RDP = np.mean([(len(ts.x_pivots) - 1) for ts in all_simplifications_RDP])
        row = ["RDP", alpha, mean_loyalty_RDP, kappa_loyalty_RDP, complexity_RDP, num_segments_RDP]
        df.loc[len(df)] = row

        # Step 1 gen all simplified ts
        logging.debug("Running BU")
        init_time = datetime.datetime.now()
        all_time_series_BU, all_simplifications_BU = get_bottom_up_simplification(time_series=all_time_series, max_error=alpha)
        time_bu.append((datetime.datetime.now()-init_time).total_seconds())

        # Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_BU]
        predicted_classes_simplifications_BU = get_model_predictions(model_path, batch_simplified_ts)
        predicted_classes_original = get_model_predictions(model_path,
                                                           all_time_series_BU)  # I will say this and all_time_series_OS are the same, but just in case

        # Step 3 calculate loyalty and complexity
        mean_loyalty_BU = calculate_mean_loyalty(pred_class_original=predicted_classes_original,
                                                  pred_class_simplified=predicted_classes_simplifications_BU)
        kappa_loyalty_BU = calculate_kappa_loyalty(pred_class_original=predicted_classes_original,
                                                    pred_class_simplified=predicted_classes_simplifications_BU)
        complexity_BU = calculate_complexity(batch_simplified_ts=all_simplifications_BU)
        num_segments_BU = np.mean([ts.num_real_segments for ts in all_simplifications_BU])
        row = ["BU", alpha, mean_loyalty_BU, kappa_loyalty_BU, complexity_BU, num_segments_BU]
        df.loc[len(df)] = row

        # Step 1 gen all simplified ts
        logging.debug("Running VC")
        init_time = datetime.datetime.now()
        all_time_series_VC, all_simplifications_VC = get_VC_simplification(time_series=all_time_series,alpha=alpha)
        time_vc.append((datetime.datetime.now()-init_time).total_seconds())

        # Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_VC]
        predicted_classes_simplifications_VC = get_model_predictions(model_path, batch_simplified_ts)
        predicted_classes_original = get_model_predictions(model_path,
                                                           all_time_series_VC)  # I will say this and all_time_series_OS are the same, but just in case

        # Step 3 calculate loyalty and complexity
        mean_loyalty_VC = calculate_mean_loyalty(pred_class_original=predicted_classes_original,
                                                 pred_class_simplified=predicted_classes_simplifications_VC)
        kappa_loyalty_VC = calculate_kappa_loyalty(pred_class_original=predicted_classes_original,
                                                   pred_class_simplified=predicted_classes_simplifications_VC)
        complexity_VC = calculate_complexity(batch_simplified_ts=all_simplifications_VC)
        num_segments_VC = np.mean([(len(ts.x_pivots) - 1) for ts in all_simplifications_VC])
        row = ["VC", alpha, mean_loyalty_VC, kappa_loyalty_VC, complexity_VC, num_segments_VC]
        df.loc[len(df)] = row

        logging.debug("Running LSF")
        init_time = datetime.datetime.now()
        all_time_series_LSF, all_simplifications_LSF = get_LSF_simplification(time_series=all_time_series, alpha=alpha)
        time_lsf = (datetime.datetime.now()-init_time).total_seconds()

        # Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_LSF]
        predicted_classes_simplifications_LSF = get_model_predictions(model_path, batch_simplified_ts)
        predicted_classes_original = get_model_predictions(model_path,
                                                           all_time_series_LSF)
        
        # Step 3 calculate loyalty and complexity
        mean_loyalty_LSF = calculate_mean_loyalty(pred_class_original=predicted_classes_original,
                                                  pred_class_simplified=predicted_classes_simplifications_LSF)
        kappa_loyalty_LSF = calculate_kappa_loyalty(pred_class_original=predicted_classes_original,
                                                    pred_class_simplified=predicted_classes_simplifications_LSF)
        complexity_LSF = calculate_complexity(batch_simplified_ts=all_simplifications_LSF)
        num_segments_LSF = np.mean([ts.num_real_segments for ts in all_simplifications_LSF])
        row = ["LSF", alpha, mean_loyalty_LSF, kappa_loyalty_LSF, complexity_LSF, num_segments_LSF]
        df.loc[len(df)] = row

    time = {"OS": np.mean(time_os), "RDP": np.mean(time_rdp), "VC": np.mean(time_vc), "BU": np.mean(time_bu), "LSF": time_lsf}

    return df, time

def get_model_predictions(model_path, batch_of_TS):
    predicted_classes = model_batch_classify(model_path, batch_of_timeseries=batch_of_TS) 
    return predicted_classes
    