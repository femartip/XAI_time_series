import numpy as np
from matplotlib import pyplot as plt
import argparse
import logging
from tqdm import tqdm
import random
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from rdp import rdp
import logging

from typing import List
from Utils.plotting import plot_csv_alpha_mean_loyalty
from ORSalgorithm.Utils.loadModel import model_batch_classify,model_classify
from Utils.load_data import load_dataset, normalize_data, get_time_series
from ORSalgorithm.simplify.DPcustomAlgoKSmallest import solve_and_find_points
from ORSalgorithm.Perturbations.dataTypes import SegmentedTS
from ORSalgorithm.Utils.scoring_functions import score_simplicity

def calculate_loyalty(batch_org_ts:List[float], batch_simplified_ts:List[SegmentedTS], model_path)->float:
    batch_simplified_ts = [ts.line_version for ts in batch_simplified_ts]
    pred_class_original = model_batch_classify(model_path, batch_of_timeseries=batch_org_ts) 
    pred_class_simplified = model_batch_classify(model_path, batch_of_timeseries=batch_simplified_ts)
    loyalty = np.mean(np.equal(pred_class_original, pred_class_simplified))
    return loyalty

def calculate_kappa_loyality(batch_org_ts:List[float], batch_simplified_ts:List[SegmentedTS], model_path)->float:
    batch_simplified_ts = [ts.line_version for ts in batch_simplified_ts]
    pred_class_original = model_batch_classify(model_path, batch_of_timeseries=batch_org_ts) 
    pred_class_simplified = model_batch_classify(model_path, batch_of_timeseries=batch_simplified_ts)
    kappa_loyalty = cohen_kappa_score(pred_class_original, pred_class_simplified)
    return kappa_loyalty

def calculate_complexity(batch_simplified_ts: List[SegmentedTS])->float:
    complexity = np.mean([score_simplicity(ts) for ts in batch_simplified_ts])
    return complexity

def simplify_all_in_dataset(dataset_name, datset_type, alpha):
    all_time_series = load_dataset(dataset_name, data_type=datset_type)
    all_simplifications = []
    first = False
    logging.debug(alpha)
    for ts_y in all_time_series:
        ts_x = [i for i in range(len(ts_y))]
        my_k = 1
        beta = 1/(len(ts_x)-1)*(1-alpha)
        selcted_xs, selectd_ys = solve_and_find_points(X=ts_x, Y=ts_y, K=my_k, alpha=alpha, beta=beta)
       
        if first:
            plt.figure()
            plt.plot(ts_x, ts_y, label="Original")
            plt.plot(selcted_xs[0], selectd_ys[0], label="Simplified")

            plt.legend()
            plt.show()
       
            first = False
        segTS = SegmentedTS(x_pivots=selcted_xs[0], y_pivots=selectd_ys[0], ts_length=len(ts_y))
        all_simplifications.append(segTS)
    
    return all_time_series, all_simplifications

def score_diff_alpha(dataset_name, datset_type, model_path):
    diff_alpha_values = np.arange(0,1,0.01)
    all_values = []
    df = pd.DataFrame(columns=["Type","Alpha", "Mean Loyalty", "Kappa Loyalty", "Complexity"])
    for alpha in tqdm(diff_alpha_values):
        # Step 1 gen all simplified ts
        all_time_seriesOS, all_simplificationsOS = simplify_all_in_dataset(dataset_name=dataset_name,datset_type=datset_type, alpha=alpha)

        # Step 2 calculate loyalty and complexity
        mean_loyaltyOS = calculate_loyalty(batch_org_ts=all_time_seriesOS, batch_simplified_ts=all_simplificationsOS, model_path=model_path)
        kappa_loyaltyOS = calculate_kappa_loyality(batch_org_ts=all_time_seriesOS, batch_simplified_ts=all_simplificationsOS, model_path=model_path)
        complexityOS = calculate_complexity(batch_simplified_ts=all_simplificationsOS)
        row = ["OS", alpha, mean_loyaltyOS, kappa_loyaltyOS, complexityOS]
        df.loc[len(df)] = row

        all_time_series_RDP, all_simplifications_RDP = get_RDP_of_timeseries(dataset_name=dataset_name, datset_type=datset_type, epsilon=alpha)
        mean_loyalty_RDP = calculate_loyalty(batch_org_ts=all_time_series_RDP, batch_simplified_ts=all_simplifications_RDP, model_path=model_path)
        kappa_loyalty_RDP = calculate_kappa_loyality(batch_org_ts=all_time_series_RDP, batch_simplified_ts=all_simplifications_RDP, model_path=model_path)
        complexity_RDP = calculate_complexity(batch_simplified_ts=all_simplifications_RDP)
        row = ["RDP", alpha, mean_loyalty_RDP, kappa_loyalty_RDP, complexity_RDP]
        df.loc[len(df)] = row

    df.to_csv(f"Alpha_complexity_loyalty_{dataset_name}.csv")


def normalize_all_dataset():
    datasets = ["Chinatown", "ECG200", "ItalyPowerDemand"]
    datatypes = ["TRAIN", "TEST", "VALIDATION"]
    for dataset in datasets:
        for datatype in datatypes:
            normalize_data(dataset, datatype)
    
def test():
    random.seed(6)
    # normalize_all_dataset()
    ts_y = get_time_series("Chinatown", "TEST", 0)
    logging.debug(ts_y)
    ts_x = [i for i in range(len(ts_y))]
    my_k = 3
    alpha = 0.1
    beta = 1/(len(ts_x)-1)*(1-alpha)
    logging.debug("Start")
    selcted_xs, selectd_ys = solve_and_find_points(X=ts_x, Y=ts_y, K=my_k, alpha=alpha, beta=beta)
    logging.debug("End")
    logging.debug(selcted_xs, selectd_ys)

    
    for i in range(len(selcted_xs)):
        plt.figure()
        plt.plot(ts_x, ts_y, label="Original")
        plt.plot(selcted_xs[i], selectd_ys[i], label="Simplified")

        plt.legend()
        plt.show()

def get_RDP_of_timeseries(dataset_name,datset_type, epsilon):
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
            plt.show()
            first = False
        all_simplifications.append(SegmentedTS(x_pivots=simp_x, y_pivots=simp_y, ts_length=len(ts_y)))
    
    return all_time_series, all_simplifications


def solve_for_alpha(dataset: str):
    dataset_name = dataset
    dataset_type = "TEST_normalized"
    model_path = f"models/{dataset}_base_norm.pth"
    score_diff_alpha(dataset_name, datset_type=dataset_type, model_path=model_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset= "ItalyPowerDemand"
    solve_for_alpha(dataset)
