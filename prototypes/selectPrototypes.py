from sklearn_extra.cluster import KMedoids
from Utils.load_data import load_dataset, load_dataset_labels
import numpy as np
from tslearn.metrics import dtw


def select_prototypes(dataset_name):
    # 1. Load a small sample dataset
    X_test = load_dataset(dataset_name=dataset_name, data_type="TEST_normalized")

    labels_test = load_dataset_labels(dataset_name=dataset_name, data_type="TEST_normalized")
    labels = np.array(labels_test)
    mask0 = labels == 0
    mask1 = labels == 1

    X_test_0 = X_test[mask0]
    X_test_1 = X_test[mask1]

    # 2. Instantiate k-medoids for 3 prototypes, using DTW distance
    km_0 = KMedoids(n_clusters=3, metric=dtw, init="random", random_state=42)

    km_1 = KMedoids(n_clusters=3, metric=dtw, init="random", random_state=42)


    # 3. Fit on your time-series array (shape: [n_samples, series_length, n_channels])
    km_0.fit(X_test_0)
    km_1.fit(X_test_1)


    # 4. Retrieve medoid indices and corresponding prototypes
    medoid_indices_0 = km_0.medoid_indices_
    medoid_indices_1 = km_1.medoid_indices_

    prototypes_0 = X_test_0[medoid_indices_0]
    prototypes_1 = X_test_1[medoid_indices_1]

    print("Selected prototype indices 0:", medoid_indices_0)
    print("Selected prototype indices 1:", medoid_indices_1)

if __name__ == "__main__":
    select_prototypes("ItalyPowerDemand")