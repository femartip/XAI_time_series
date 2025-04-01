from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from interpret.utils import SPOT_GreedySubsetSelection  # This loads the SPOT prototype selection algorithm.

from utils.data import load_dataset_ts
from models.loadModel import model_batch_classify


def get_class_protos(org_idx, x_data, title=""):
    # Split data into 70% src and 30% tgt subsets
    # X_src, X_tgt, y_src, y_tgt = train_test_split(
    #    x_data, y_data, test_size=0.3, shuffle=False)

    # Compute the Euclidean distances between the X_src (source) and X_tgt (target) points.
    C = pairwise_distances(x_data, x_data, metric='euclidean')
    # Define a targetmarginal on the target set
    # We define the uniform marginal
    targetmarginal = np.ones(C.shape[1]) / C.shape[1]

    rows = 1
    cols = 3
    # The number of prototypes to be computed
    numprototypes = rows * cols

    # Run SPOTgreedy
    # prototypeIndices represent the indices corresponding to the chosen prototypes.
    # prototypeWeights represent the weights associated with each of the chosen prototypes. The weights sum to 1.
    [prototypeIndices, prototypeWeights] = SPOT_GreedySubsetSelection(C, targetmarginal, numprototypes)

    return [org_idx[idx] for idx in prototypeIndices]


def get_prototypes(dataset_name: str, model_name: str, verbose: bool = False):
    # Load the digits dataset
    # digits = load_digits()
    test_data = load_dataset_ts(dataset_name=dataset_name, data_type="TEST")
    print(test_data[0].shape)

    pred_class = np.array(model_batch_classify(model_name, test_data))
    org_idx = np.array(list(range(len(pred_class))))
    org_idx_class_0 = org_idx[pred_class == 0]
    x_data_class_0 = test_data[org_idx_class_0]
    # y_data_class_0 = test_data[org_idx_class_0]
    selected_idx_class_0 = get_class_protos(org_idx_class_0, x_data_class_0, title=f"{dataset_name}, class {0}")

    org_idx_class_1 = org_idx[pred_class == 1]
    x_data_class_1 = test_data[org_idx_class_1]
    # y_data_class_1 = pred_class[org_idx_class_1]
    selected_idx_class_1 = get_class_protos(org_idx_class_1, x_data_class_1, title=f"{dataset_name}, class {1}")
    rows = 2
    cols = 3
    if verbose:
        fig, ax = plt.subplots(rows, cols)

        if rows == 1 or cols == 1:
            ax = np.array(ax).reshape(rows, cols)

        selected_idx = selected_idx_class_0 + selected_idx_class_1
        for i, idx in enumerate(selected_idx):
            row = i // cols
            col = i % cols
            ax[row][col].plot(list(range(len(test_data[idx]))), test_data[idx])
        plt.suptitle(f"Both classes of {dataset_name}")
        plt.tight_layout()
        plt.show()

    return selected_idx_class_0, selected_idx_class_1


def all_datasets():
    dataset_names = ["ItalyPowerDemand", "Chinatown", "ECG200"]
    model_names = ["ItalyPowerDemand_100.keras", "Chinatown_100.keras", "ECG200_100.keras"]
    for dataset_name, model_name in zip(dataset_names, model_names):
        get_prototypes(dataset_name=dataset_name, model_name=model_name, verbose=True)


if __name__ == '__main__':
    all_datasets()
