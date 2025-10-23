import numpy as np
from Utils.load_data import load_dataset
from Utils.load_models import model_batch_classify

import hashlib
from random import Random
import warnings
warnings.simplefilter("always")
def stable_random_for(dataset_name: str) -> Random:
    """
    To increase randomness we base the selection on the dataset name.
    We use hashlib to have a STABLE hash, that will always stay the same.
    """
    seed = int(hashlib.sha256(dataset_name.encode()).hexdigest(), 16) % (2**32)
    return Random(seed)


def select_test_examples(dataset_name: str):
    model_path = f"models/{dataset_name}/miniRocket.pkl"

    # 1. Load dataset and labels
    X_test = load_dataset(dataset_name=dataset_name, data_type="TEST")
    labels_test = model_batch_classify(model_path=model_path, batch_of_timeseries=X_test, num_classes=2)
    labels = np.array(labels_test)

    unique_labels = np.unique(labels)

    # 2. For each label, randomly sample instances
    num_instance_per_class = 5
    label_to_test = {}

    for label in unique_labels:
        mask = labels == label # Correct pred label and not training instance
        X_label = np.where(mask)[0].tolist()
        valid_instances = len(X_label)

        assert valid_instances>= num_instance_per_class
        myRand = stable_random_for(dataset_name=dataset_name)
        selected_instance = myRand.sample(X_label,k=num_instance_per_class)

        label_to_test[label] = selected_instance
    # 3. Concatenate all prototypes
    return label_to_test

if __name__ == "__main__":
    print(select_test_examples("Chinatown", list(range(1,230))))

