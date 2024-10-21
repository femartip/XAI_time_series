import numpy as np
from matplotlib import pyplot as plt
import argparse

from Utils.load_data import load_dataset

from ORSalgorithm.ORSalgorithm import ORSalgorithm

def generate_approximation_ts_for_all_in_dataset(dataset_name, model_path, my_k, alpha):
    all_time_series = load_dataset(dataset_name, data_type="TEST")
    ORSalgorithm(all_time_series, model_path, k=my_k, alpha=alpha)
            

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_name", type=str, default="Chinatown", help="Name of the dataset, can be either Chinatown, ECG200 or ItalyEnergy")
    args.add_argument("--model_path", type=str, default="models/ORSmodel", help="Path to the pytorch model")
    args.add_argument("--k", type=int, default=1, help="Number of k best solutions")
    args.add_argument("--alpha", type=float, default=0.2)
    args = args.parse_args()
    
    generate_approximation_ts_for_all_in_dataset(args.dataset_name, args.model_path, args.k, args.alpha)
