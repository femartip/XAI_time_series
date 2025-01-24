import logging
from evaluation import score_different_alphas
from Utils.plotting import plot_csv_complexity_kappa_loyalty, plot_csv_alpha_mean_loyalty
import argparse
import os

def main(dataset: str, dataset_type: str, model_type: str):
    """
    Main function to evaluate simplifications.
    """
    if model_type == "cnn":
        model_path = f"models/{dataset}_cnn_norm.pth"
    else: 
        model_path = f"models/{dataset}_{model_type}_norm.pkl"
    score_different_alphas(dataset, datset_type=dataset_type, model_path=model_path)
    if os.path.exists(f"results/{model_type}/Alpha_complexity_loyalty_{dataset}.csv"):
        plot_csv_alpha_mean_loyalty(f"results/{model_type}/Alpha_complexity_loyalty_{dataset}.csv")
        plot_csv_complexity_kappa_loyalty(f"results/{model_type}/Alpha_complexity_loyalty_{dataset}.csv")
    else:
        logging.error("Results not saved to CSV.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate simplifications')
    parser.add_argument('--dataset', type=str, default="Chinatown", help='Dataset name, for now Chinatown, ECG200 and ItalyPowerDemand supported')
    parser.add_argument('--dataset_type', type=str, default="TEST_normalized", help='Dataset type, can be either TRAIN, TEST, VALIDATION with or without _normalized')
    parser.add_argument('--model_type', type=str, default="cnn", help='Model type, can be either cnn, decision-tree, knn')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info("Selected configuration:")
    logging.info(args)
    main(dataset=args.dataset, dataset_type=args.dataset_type, model_type=args.model_type)
