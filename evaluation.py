import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from simplifications import get_OS_simplification, get_RDP_simplification
from Utils.metrics import calculate_accuracy_loyalty, calculate_kappa_loyalty, calculate_complexity


def score_different_alphas(dataset_name, datset_type, model_path):
    """
    Evaluate the impact of different alpha values on loyalty, kappa, and complexity.
    """
    diff_alpha_values = np.arange(0,1,0.1)
    df = pd.DataFrame(columns=["Type","Alpha", "Mean Loyalty", "Kappa Loyalty", "Complexity"])

    for alpha in tqdm(diff_alpha_values):
        # Step 1 gen all simplified ts
        all_time_series_OS, all_simplificationsOS = get_OS_simplification(dataset_name=dataset_name,datset_type=datset_type, alpha=alpha)

        # Step 2 calculate loyalty and complexity
        mean_loyaltyOS = calculate_accuracy_loyalty(batch_org_ts=all_time_series_OS, batch_simplified_ts=all_simplificationsOS, model_path=model_path)
        kappa_loyaltyOS = calculate_kappa_loyalty(batch_org_ts=all_time_series_OS, batch_simplified_ts=all_simplificationsOS, model_path=model_path)
        complexityOS = calculate_complexity(batch_simplified_ts=all_simplificationsOS)
        row = ["OS", alpha, mean_loyaltyOS, kappa_loyaltyOS, complexityOS]
        df.loc[len(df)] = row

        all_time_series_RDP, all_simplifications_RDP = get_RDP_simplification(dataset_name=dataset_name, datset_type=datset_type, epsilon=alpha)
        mean_loyalty_RDP = calculate_accuracy_loyalty(batch_org_ts=all_time_series_RDP, batch_simplified_ts=all_simplifications_RDP, model_path=model_path)
        kappa_loyalty_RDP = calculate_kappa_loyalty(batch_org_ts=all_time_series_RDP, batch_simplified_ts=all_simplifications_RDP, model_path=model_path)
        complexity_RDP = calculate_complexity(batch_simplified_ts=all_simplifications_RDP)
        row = ["RDP", alpha, mean_loyalty_RDP, kappa_loyalty_RDP, complexity_RDP]
        df.loc[len(df)] = row

    df.to_csv(f"Alpha_complexity_loyalty_{dataset_name}.csv")
    logging.info("Results saved to CSV.")