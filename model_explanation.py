import torch
import argparse
import logging

from Utils.load_data import load_dataset
from ORSalgorithm.ORSalgorithm import ORSalgorithm

def explain_model(dataset_name: str, blackbox_model_path: str):
    time_series = load_dataset(dataset_name, data_type="TRAIN")

    for ts in time_series:
        prototype, confidence = ORSalgorithm(ts, blackbox_model_path, k=1, alpha=0.02)      #Extract prototype from blackbox model

        


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_name", type=str, default="Chinatown", help="Name of the dataset, can be either Chinatown, ECG200 or ItalyEnergy")
    args.add_argument("--blackbox_model_path", type=str, default="models/ORSmodel", help="Path to the pytorch model")

    args = args.parse_args()

    explain_model(args.dataset_name, args.blackbox_model_path)
