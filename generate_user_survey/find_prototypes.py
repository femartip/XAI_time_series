import numpy as np

from Utils.load_data import load_dataset
from Utils.load_models import model_batch_classify, model_classify

from sklearn_extra.cluster import KMedoids
from tslearn.metrics import dtw

from typing import Dict, List

import warnings
warnings.simplefilter("always")
def select_prototypes(dataset_name: str) -> Dict[str, List[int]] :
    # 1. Load dataset and labels
    model_path = f"models/{dataset_name}/miniRocket.pkl"
    num_instances = 3

    X_train = load_dataset(dataset_name=dataset_name, data_type="TRAIN")
    labels_test = model_batch_classify(model_path=model_path,batch_of_timeseries=X_train,num_classes=2)
    labels = np.array(labels_test)

    unique_labels = np.unique(labels)

    # 2. For each label, fit KMedoids and collect prototypes
    label_to_prototypes = {}
    for label in unique_labels:
        mask = labels == label
        X_label = X_train[mask]
        assert len(X_label) >= num_instances

        km = KMedoids(n_clusters=num_instances, metric=dtw, init="random", random_state=42)  # type: ignore
        km.fit(X_label)
        medoid_indices_local = km.medoid_indices_
        medoid_indices_global = np.where(mask)[0][medoid_indices_local]
        label_to_prototypes[label] = medoid_indices_global

    # 3. Concatenate all prototypes
    return label_to_prototypes

if __name__ == "__main__":
    print(select_prototypes("PhalangesOutlinesCorrect"))
    print("Done")

