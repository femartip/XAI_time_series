import logging
from evaluation import score_different_alphas
from Utils.plotting import plot_csv_complexity_kappa_loyalty, plot_csv_alpha_mean_loyalty, plot_csv_complexity_mean_loyalty
import argparse
import os
from train_models import train_model, save_model
import pandas as pd

def main(dataset: str, dataset_type: str, model_type: str):
    """
    Main function to evaluate simplifications.
    Checks that requested model exists, if not trains it.
    Evaluates model on different alphas and saves results to CSV.
    Saves plots of results.
    """
    normalized = True if "normalized" in dataset_type else False

    if not os.path.exists(f"results/{dataset}"):
        os.makedirs(f"results/{dataset}")

    if not os.path.exists(f"models/{dataset}"):
        os.makedirs(f"models/{dataset}")

    model_csv = f"results/{dataset}/models.csv"

    if model_type == "cnn":
        model_path = f"models/{dataset}/cnn_norm.pth" if normalized else f"models/{dataset}/cnn.pth"
        if not os.path.exists(model_path):
            model, metrics = train_model(dataset, model_type, normalized=normalized)  
            if os.path.exists(model_csv):
                model_df = pd.read_csv(model_csv, header=0)
            else:
                model_df = pd.DataFrame(columns=["model_type", "train_acc", "val_acc", "test_acc"])

            model_df.loc[len(model_df)] = [model_type, metrics["train_acc"], metrics["val_acc"], metrics["test_acc"]]
            model_df.to_csv(model_csv, index=False)
            save_model(model, model_path, model_type)
    else: 
        model_path = f"models/{dataset}/{model_type}_norm.pkl" if normalized else f"models/{dataset}/{model_type}.pkl"
        if not os.path.exists(model_path):
            model, metrics = train_model(dataset, model_type, normalized=normalized)  
            if os.path.exists(model_csv):
                model_df = pd.read_csv(model_csv, header=0)
            else:
                model_df = pd.DataFrame(columns=["model_type", "train_acc", "val_acc", "test_acc"])

            model_df.loc[len(model_df)] = [model_type, metrics["train_acc"], metrics["val_acc"], metrics["test_acc"]]
            model_df.to_csv(model_csv, index=False)
            save_model(model, model_path, model_type)

    df = score_different_alphas(dataset, datset_type=dataset_type, model_path=model_path)
    df.to_csv(f"results/{dataset}/{model_type}_alpha_complexity_loyalty.csv", index=False)

    if os.path.exists(f"results/{dataset}/{model_type}_alpha_complexity_loyalty.csv"):
        output_file = f"results/{dataset}/{model_type}_alpha_complexity_loyalty.csv"
        fig1 = plot_csv_alpha_mean_loyalty(output_file)
        fig1.savefig(f"results/{dataset}/{model_type}_alpha_mean_loyalty.png")

        fig2 = plot_csv_complexity_kappa_loyalty(output_file)
        fig2.savefig(f"results/{dataset}/{model_type}_complexity_kappa_loyalty.png")

        fig3 = plot_csv_complexity_mean_loyalty(output_file)
        fig3.savefig(f"results/{dataset}/{model_type}_complexity_mean_loyalty.png")
    else:
        logging.error("Results not saved to CSV.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate simplifications')
    parser.add_argument('--datasets', type=str, nargs='+', help='List of dataset names, if not specified will evaluate all datasets in data folder.')
    parser.add_argument('--dataset_type', type=str, default="TEST_normalized", help='Dataset type, can be either TRAIN, TEST, VALIDATION with or without _normalized')
    parser.add_argument('--model_type', type=str, help='Model type, can be either cnn, decision-tree, knn. If not specified will use all models.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    if args.datasets is not None:
        datasets = args.datasets
    else:
        datasets = [x for x in os.listdir("./data/") if os.path.isdir(f"./data/{x}")]
    if args.model_type is not None:
        model_types = [args.model_type]
    else:     
        model_types = ["cnn", "decision-tree", "logistic-regression", "knn"]

    for dataset in datasets:
        for model_type in model_types:
            logging.info("Configuration:")
            logging.info(f"Dataset: {dataset}, Dataset Type: {args.dataset_type}, Model Type: {model_type}")
            main(dataset=dataset, dataset_type=args.dataset_type, model_type=model_type)
