import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import datetime
import os
import random
from sklearn.model_selection import train_test_split
import re

from simplifications import get_OS_simplification, get_RDP_simplification, get_bottom_up_simplification, \
    get_VW_simplification, get_LSF_simplification
from Utils.metrics import calculate_mean_loyalty, calculate_kappa_loyalty, calculate_complexity, score_simplicity, calculate_percentage_agreement
from Utils.load_models import model_batch_classify # type: ignore
from Utils.load_data import load_dataset, load_dataset_labels
from Utils.dataTypes import SegmentedTS 

logging.basicConfig(level=logging.debug) #type: ignore


def score_different_alphas(dataset_name: str, datset_type: str, model_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate the impact of different alpha values on loyalty, kappa, and complexity.
    """
    clear_saved_simplifications(dataset_name, datset_type, model_path)

    diff_alpha_values = np.arange(0,1,0.01)     #Bigger steps of alpha skew the results
    df = pd.DataFrame(columns=["Type","Alpha", "Percentage Agreement", "Kappa Loyalty", "Complexity", "Num Segments"])
    all_time_series = load_dataset(dataset_name, data_type=datset_type)
    labels = load_dataset_labels(dataset_name, data_type=datset_type)
    
    # Algorithms grow in time complexity with the number of time series. If there are too many, we will stratify the data to get 100 samples
    if np.shape(all_time_series)[0] > 100:
        real_shape = np.shape(all_time_series)
        porcentage_data = 100/real_shape[0]
        all_time_series, _ = train_test_split(all_time_series, train_size=porcentage_data, stratify=labels, random_state=42)
        logging.info(f"Number of instances {real_shape} > 100, so modified to: {np.shape(all_time_series)[0]}")

    predicted_classes_original = get_model_predictions(model_path, all_time_series) #type: ignore

    time_os = []
    time_rdp = []
    time_vw = []
    time_bu = []

    for alpha in tqdm(diff_alpha_values):
        # Step 1 gen all simplified ts
        logging.debug(f"Alpha: {alpha}")

        logging.debug("Running OS")
        init_time = datetime.datetime.now()
        all_simplifications_OS = get_OS_simplification(time_series=all_time_series, alpha=alpha)
        time_os.append((datetime.datetime.now()-init_time).total_seconds())

        #Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_OS]
        predicted_classes_simplifications_OS = get_model_predictions(model_path, batch_simplified_ts) 

        # Step 3 calculate loyalty and complexity
        #mean_loyalty_OS = calculate_mean_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_OS)
        kappa_loyalty_OS = calculate_kappa_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_OS)
        percentage_agreement_OS = calculate_percentage_agreement(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_OS)
        complexity_OS = calculate_complexity(batch_simplified_ts=all_simplifications_OS)
        num_segments_OS = np.mean([(len(ts.x_pivots) - 1) for ts in all_simplifications_OS])
        row = ["OS", alpha, percentage_agreement_OS, kappa_loyalty_OS, complexity_OS, num_segments_OS]
        df.loc[len(df)] = row

        save_simplifications(os_alg="OS", dataset_name=dataset_name, dataset_type=datset_type, model_path=model_path, X=batch_simplified_ts, classes=predicted_classes_simplifications_OS, alpha=alpha)

        # Step 1 gen all simplified ts
        logging.debug("Running RDP")
        init_time = datetime.datetime.now()
        all_simplifications_RDP = get_RDP_simplification(time_series=all_time_series, epsilon=alpha)
        time_rdp.append((datetime.datetime.now()-init_time).total_seconds())

        #Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_RDP]
        predicted_classes_simplifications_RDP = get_model_predictions(model_path, batch_simplified_ts)   

        # Step 3 calculate loyalty and complexity
        #mean_loyalty_RDP = calculate_mean_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_RDP)
        kappa_loyalty_RDP = calculate_kappa_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_RDP)
        percentage_agreement_RDP = calculate_percentage_agreement(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_RDP)
        complexity_RDP = calculate_complexity(batch_simplified_ts=all_simplifications_RDP)
        num_segments_RDP = np.mean([(len(ts.x_pivots) - 1) for ts in all_simplifications_RDP])
        row = ["RDP", alpha, percentage_agreement_RDP, kappa_loyalty_RDP, complexity_RDP, num_segments_RDP]
        df.loc[len(df)] = row

        save_simplifications(os_alg="RDP", dataset_name=dataset_name, dataset_type=datset_type, model_path=model_path, X=batch_simplified_ts, classes=predicted_classes_simplifications_RDP, alpha=alpha)

        # Step 1 gen all simplified ts
        logging.debug("Running BU")
        init_time = datetime.datetime.now()
        all_simplifications_BU = get_bottom_up_simplification(time_series=all_time_series, max_error=alpha)
        time_bu.append((datetime.datetime.now()-init_time).total_seconds())

        # Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_BU]
        predicted_classes_simplifications_BU = get_model_predictions(model_path, batch_simplified_ts)  #type: ignore I will say this and all_time_series_OS are the same, but just in case

        # Step 3 calculate loyalty and complexity
        #mean_loyalty_BU = calculate_mean_loyalty(pred_class_original=predicted_classes_original,pred_class_simplified=predicted_classes_simplifications_BU)
        kappa_loyalty_BU = calculate_kappa_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_BU)
        percentage_agreement_BU = calculate_percentage_agreement(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_BU)
        complexity_BU = calculate_complexity(batch_simplified_ts=all_simplifications_BU)
        num_segments_BU = np.mean([ts.num_real_segments for ts in all_simplifications_BU])
        row = ["BU", alpha, percentage_agreement_BU, kappa_loyalty_BU, complexity_BU, num_segments_BU]
        df.loc[len(df)] = row

        save_simplifications(os_alg="BU", dataset_name=dataset_name, dataset_type=datset_type, model_path=model_path, X=batch_simplified_ts, classes=predicted_classes_simplifications_BU, alpha=alpha)

        # Step 1 gen all simplified ts
        logging.debug("Running VW")
        init_time = datetime.datetime.now()
        all_simplifications_VW = get_VW_simplification(time_series=all_time_series,alpha=alpha)
        time_vw.append((datetime.datetime.now()-init_time).total_seconds())

        # Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_VW]
        predicted_classes_simplifications_VW = get_model_predictions(model_path, batch_simplified_ts)  #type: ignore I will say this and all_time_series_OS are the same, but just in case

        # Step 3 calculate loyalty and complexity
        #mean_loyalty_VW = calculate_mean_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_VW)
        kappa_loyalty_VW = calculate_kappa_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_VW)
        percentage_agreement_VW = calculate_percentage_agreement(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_VW)
        complexity_VW = calculate_complexity(batch_simplified_ts=all_simplifications_VW)
        num_segments_VW = np.mean([(len(ts.x_pivots) - 1) for ts in all_simplifications_VW])
        row = ["VW", alpha, percentage_agreement_VW, kappa_loyalty_VW, complexity_VW, num_segments_VW]
        df.loc[len(df)] = row

        save_simplifications(os_alg="VW", dataset_name=dataset_name, dataset_type=datset_type, model_path=model_path, X=batch_simplified_ts, classes=predicted_classes_simplifications_VW, alpha=alpha)

        """
        logging.debug("Running LSF")
        init_time = datetime.datetime.now()
        all_simplifications_LSF = get_LSF_simplification(time_series=all_time_series, alpha=alpha)
        time_lsf = (datetime.datetime.now()-init_time).total_seconds()

        # Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_LSF]
        predicted_classes_simplifications_LSF = get_model_predictions(model_path, batch_simplified_ts)
        
        # Step 3 calculate loyalty and complexity
        mean_loyalty_LSF = calculate_mean_loyalty(pred_class_original=predicted_classes_original,
                                                  pred_class_simplified=predicted_classes_simplifications_LSF)
        kappa_loyalty_LSF = calculate_kappa_loyalty(pred_class_original=predicted_classes_original,
                                                    pred_class_simplified=predicted_classes_simplifications_LSF)
        complexity_LSF = calculate_complexity(batch_simplified_ts=all_simplifications_LSF)
        num_segments_LSF = np.mean([ts.num_real_segments for ts in all_simplifications_LSF])
        row = ["LSF", alpha, mean_loyalty_LSF, kappa_loyalty_LSF, complexity_LSF, num_segments_LSF]
        """
        row = ["LSF", alpha, 0, 0, 0, 0]
        time_lsf = 0
        df.loc[len(df)] = row

    time = {"OS": np.mean(time_os), "RDP": np.mean(time_rdp), "VW": np.mean(time_vw), "BU": np.mean(time_bu), "LSF": time_lsf}

    return df, time

def get_model_predictions(model_path: str, batch_of_TS: list[list[float]]) -> list[int]:
    predicted_classes = model_batch_classify(model_path, batch_of_timeseries=batch_of_TS) 
    return predicted_classes

def clear_saved_simplifications(dataset_name: str, dataset_type: str, model_path: str) -> None:
    dataset_path = f"results/{dataset_name}/data"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    model = model_path.split("/")[-1].split("_")[0]
    files = os.listdir(dataset_path)
    for file in files:
        if re.match(rf"\w+_{model}_{dataset_type}\.npy", file):
            file_path = os.path.join(dataset_path, file)
            os.remove(file_path)
            logging.info(f"Removed file {file_path}")
    
def save_simplifications(os_alg: str, dataset_name: str, dataset_type: str, model_path: str, X: list[list[float]], classes: list[int], alpha: float) -> None:
    num_instances = len(X)
    dim = len(X[0])  
    dtype = np.dtype([('alpha', np.float64, (1,)), ('class', np.int8, (1,)),('X', np.float64, (dim,))])
    alphas = np.array([alpha] * len(classes))[:, np.newaxis]
    classes = np.array(classes, np.int8)[:, np.newaxis]     #type: ignore
    X = np.array(X)     #type: ignore

    combined_data = np.zeros(num_instances, dtype=dtype)
    combined_data['alpha'] = alphas
    combined_data['class'] = classes
    combined_data['X'] = X

    model = model_path.split("/")[-1].split("_")[0]
    file_path = f"results/{dataset_name}/data/{os_alg}_{model}_{dataset_type}.npy"
    if not os.path.exists(file_path):
        np.save(file_path, combined_data, allow_pickle=False)
    else:
        data = np.load(file_path, allow_pickle=False)
        data = np.append(data, combined_data, axis=0)
        np.save(file_path, data, allow_pickle=False)
    