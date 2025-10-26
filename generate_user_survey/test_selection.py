import hashlib
import warnings
from random import Random

import numpy as np

from Utils.load_data import load_dataset
from Utils.load_models import model_batch_classify

warnings.simplefilter("always")


def stable_random_for(dataset_name: str) -> Random:
    """
    To increase randomness we base the selection on the dataset name.
    We use hashlib to have a STABLE hash, that will always stay the same.
    """
    seed = int(hashlib.sha256(dataset_name.encode()).hexdigest(), 16) % (2 ** 32)
    return Random(seed)


def distribute_test_samples(k, total=10):
    base = total // k
    remainder = total % k
    distribution = [base + 1] * remainder + [base] * (k - remainder)
    return distribution


def select_test_examples(dataset_name: str):
    model_path = f"models/{dataset_name}/miniRocket.pkl"

    # 1. Load dataset and labels
    X_test = load_dataset(dataset_name=dataset_name, data_type="TEST")
    labels_test = model_batch_classify(model_path=model_path, batch_of_timeseries=X_test, num_classes=2)
    labels = np.array(labels_test)

    unique_labels = sorted(np.unique(labels).tolist())

    # 2. For each label, randomly sample instances
    distribution = distribute_test_samples(len(unique_labels), total=10)
    label_to_test = {}

    for i, label in enumerate(unique_labels):
        mask = labels == label  # Correct pred label and not training instance
        X_label = np.where(mask)[0].tolist()
        valid_instances = len(X_label)

        assert valid_instances >= distribution[i]
        myRand = stable_random_for(dataset_name=dataset_name)
        selected_instance = myRand.sample(X_label, k=distribution[i])

        label_to_test[label] = selected_instance
    # 3. Concatenate all prototypes
    return label_to_test


if __name__ == "__main__":
    print(select_test_examples("UMD"))
