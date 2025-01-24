import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import os

from simplifications import get_OS_simplification, get_RDP_simplification
from Utils.metrics import calculate_mean_loyalty, calculate_kappa_loyalty, calculate_complexity
from Utils.load_models import model_batch_classify # type: ignore


def score_different_alphas(dataset_name, datset_type, model_path):
    """
    Evaluate the impact of different alpha values on loyalty, kappa, and complexity.
    """
    diff_alpha_values = np.arange(0,1,0.01)
    df = pd.DataFrame(columns=["Type","Alpha", "Mean Loyalty", "Kappa Loyalty", "Complexity"])

    for alpha in tqdm(diff_alpha_values):
        # Step 1 gen all simplified ts
        all_time_series_OS, all_simplificationsOS = get_OS_simplification(dataset_name=dataset_name,datset_type=datset_type, alpha=alpha)

        #Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplificationsOS]
        predicted_classes_simplifications_OS = get_model_predictions(model_path, batch_simplified_ts)
        predicted_classes_original = get_model_predictions(model_path, all_time_series_OS)

        # Step 3 calculate loyalty and complexity
        mean_loyaltyOS = calculate_mean_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_OS)
        kappa_loyaltyOS = calculate_kappa_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_OS)
        complexityOS = calculate_complexity(batch_simplified_ts=all_simplificationsOS)
        row = ["OS", alpha, mean_loyaltyOS, kappa_loyaltyOS, complexityOS]
        df.loc[len(df)] = row

        # Step 1 gen all simplified ts
        all_time_series_RDP, all_simplifications_RDP = get_RDP_simplification(dataset_name=dataset_name, datset_type=datset_type, epsilon=alpha)
        
        #Step 2 get model predictions
        batch_simplified_ts = [ts.line_version for ts in all_simplifications_RDP]
        predicted_classes_simplifications_RDP = get_model_predictions(model_path, batch_simplified_ts)
        predicted_classes_original = get_model_predictions(model_path, all_time_series_RDP)     #I will say this and all_time_series_OS are the same, but just in case

        # Step 3 calculate loyalty and complexity
        mean_loyalty_RDP = calculate_mean_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_RDP)
        kappa_loyalty_RDP = calculate_kappa_loyalty(pred_class_original=predicted_classes_original, pred_class_simplified=predicted_classes_simplifications_RDP)
        complexity_RDP = calculate_complexity(batch_simplified_ts=all_simplifications_RDP)
        row = ["RDP", alpha, mean_loyalty_RDP, kappa_loyalty_RDP, complexity_RDP]
        df.loc[len(df)] = row

    model_type = model_path.split("_")[1]
    if os.path.exists(f"results/{model_type}"):
        df.to_csv(f"results/{model_type}/Alpha_complexity_loyalty_{dataset_name}.csv")
        logging.info("Results saved to CSV.")
    else:
        logging.error("Results not saved to CSV.")

def get_model_predictions(model_path, batch_of_TS):
    predicted_classes = model_batch_classify(model_path, batch_of_timeseries=batch_of_TS) 
    return predicted_classes
    