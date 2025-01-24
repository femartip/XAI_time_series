import logging
from evaluation import score_different_alphas
from Utils.plotting import plot_csv_complexity_kappa_loyalty

def main(dataset: str):
    """
    Main function to evaluate simplifications.
    """
    dataset_name = dataset
    dataset_type = "TEST_normalized"
    model_path = f"models/{dataset}_cnn_norm.pth"
    score_different_alphas(dataset_name, datset_type=dataset_type, model_path=model_path)
    plot_csv_complexity_kappa_loyalty(f"Alpha_complexity_loyalty_{dataset_name}.csv")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset= "Chinatown"
    main(dataset)
