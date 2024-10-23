import numpy as np
from matplotlib import pyplot as plt
import argparse
import logging

from Utils.load_data import load_dataset
from ORSalgorithm.ORSalgorithm import ORSalgorithm

def generate_prototypes_for_best_alpha(dataset_name, model_path, my_k, show_plots=False):
    all_time_series = load_dataset(dataset_name, data_type="TRAIN")
    best_alphas = []
    best_confidences = []
    best_simplifications = []

    for ts in all_time_series:
        best_alpha = None
        best_confidence = -np.inf
        best_simplification = None

        for a in np.arange(0.1, 1.1, 0.1):
            simplification, confidence = ORSalgorithm(np.array([ts]), model_path, k=my_k, alpha=a)
            avg_confidence = np.mean(confidence)
            if avg_confidence > best_confidence:
                best_confidence = avg_confidence
                best_alpha = a
                best_simplification = simplification

        best_alphas.append(best_alpha)
        best_confidences.append(best_confidence)
        best_simplifications.append(best_simplification)

    if show_plots:
        for i in range(len(all_time_series)):
            plt.figure()
            plt.plot(all_time_series[i], label="Original")
            plt.plot(best_simplifications[i][0], label="Simplified")
            plt.title("Time series " + str(i) + " with best alpha=" + str(best_alphas[i]) + " and confidence=" + str(best_confidences[i]))
            plt.legend()
            plt.show()
            

    return best_alphas, best_confidences, best_simplifications

    
def generate_prototypes(dataset_name, model_path, k, alpha):
    all_time_series = load_dataset(dataset_name, data_type="TRAIN")

    simplification, confidence = ORSalgorithm(all_time_series, model_path, k=k, alpha=alpha)
    
    for i in range(len(all_time_series)):
        plt.figure()
        plt.plot(all_time_series[i], label="Original")
        plt.plot(simplification[i], label="Simplified")
        plt.title("Time series " + str(i) + " with alpha=" + str(alpha) + " and confidence=" + str(confidence[i]))
        plt.legend()
        plt.show()

    return simplification, confidence

    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_name", type=str, default="Chinatown", help="Name of the dataset, can be either Chinatown, ECG200 or ItalyEnergy")
    args.add_argument("--model_path", type=str, default="models/ORSmodel", help="Path to the pytorch model")
    args.add_argument("--k", type=int, default=1, help="Number of k best solutions")
    args.add_argument("--alpha", type=float, default=0.02, help="Tunable parameter that controls the fit of the simplified prototype to the original time series")
    args = args.parse_args()
    
    #generate_prototypes_for_best_alpha(args.dataset_name, args.model_path, args.k)
    generate_prototypes(args.dataset_name, args.model_path, args.k, args.alpha)
