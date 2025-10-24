from collections import defaultdict

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from Utils.load_data import load_dataset


def get_complexity_at_loyalty_dataset(dataset_name,algo, model_name, loyalty_threshold):
    df = pd.read_csv(f"results/{dataset_name}/{model_name}_alpha_complexity_loyalty.csv")
    df = df[df["Type"] == algo]
    df = df[df["Kappa Loyalty"] >= loyalty_threshold/100]
    min_complexity = np.min(df["Complexity"])
    return min_complexity

def get_complexity_at_loyalty(dataset_names, algo,model_name, loyalty_threshold):
    lt_dict = defaultdict(list)
    for dataset_name in dataset_names:
        complexity = get_complexity_at_loyalty_dataset(dataset_name, algo, model_name, loyalty_threshold)
        length = len(load_dataset(dataset_name=dataset_name)[0])
        lt_dict[length].append(complexity)

    lengths = []
    complexities = []
    for length in sorted(lt_dict.keys()):
        lengths.append(length)
        complexities.append(np.mean(lt_dict[length]))

    return lengths, complexities

def plot_complexity_length_at_loyalty(dataset_names, algo,model_name, loyalty_thresholds):
    lt_threshold = {}
    min_length = float("inf")
    max_length = float("-inf")
    for loyalty_threshold in loyalty_thresholds:
        lengths, complexities = get_complexity_at_loyalty(dataset_names,algo, model_name, loyalty_threshold)
        lt_threshold[loyalty_threshold] = (lengths, complexities)
        min_length = min(min(lengths), min_length)
        max_length = max(max(lengths), max_length)

    for loyalty_threshold in loyalty_thresholds:
        plt.plot(*lt_threshold[loyalty_threshold],label=f"{algo} by >= {loyalty_threshold}")

    baseline = 10
    x = np.linspace(min_length,max_length,100)
    y = baseline/x
    plt.plot(x,y,'--', label=f"{baseline} Segments")

    plt.legend(loc="upper right")
    plt.xlabel("Length")
    plt.ylabel("Complexity")
    plt.title(f"Complexity vs. Length {algo}")
    plt.ylim((0,1))

    folder_name = "PivotEval/Complexity_vs_length"
    os.makedirs(folder_name, exist_ok=True)
    plt.savefig(folder_name + f"/Complexity_vs_length_{algo}.png")
    plt.show()


if __name__ == "__main__":
    dataset_names = [dataset for dataset in os.listdir("results") if os.path.isdir(f"results/{dataset}")]
    model= "miniRocket"
    algos = ["OS", "RDP", "VW"]
    loyalty_thresholds = [80,90,95,100]
    for algo in algos:
        plot_complexity_length_at_loyalty(dataset_names=dataset_names, algo=algo, model_name=model,loyalty_thresholds=loyalty_thresholds)




