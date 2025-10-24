import numpy
import numpy as np
import pandas as pd
from Utils.load_models import load_model
from Utils.load_data import load_dataset, load_dataset_labels
import shap

import torch

from matplotlib import pyplot as plt

import torch


def get_classes(model_name, dataset_name, instances_to_be_explained):
    model_path = f"models/{dataset_name}/{model_name}"
    labels = load_dataset_labels(dataset_name=dataset_name)
    num_classes = len(set(labels))
    model = load_model(model_path, num_classes)
    if model_name.endswith(".pth"):
        model.eval()

        instance_processed = torch.tensor(instances_to_be_explained.reshape(1, 1, len(instance_to_be_explained)),
                                      dtype=torch.float32)

        proba =  model(instance_processed)[0].detach().numpy()
        return np.argmax(proba)
    else:
        pred_class = model.predict(np.array(instances_to_be_explained))
        return pred_class

def get_shap_values(model_name, background, dataset_name, instances_to_be_explained):
    model_path = f"models/{dataset_name}/{model_name}"
    labels = load_dataset_labels(dataset_name=dataset_name)
    num_classes = len(set(labels))
    model_loaded = load_model(model_path, num_classes)
    if model_name.endswith(".pth"):
        model = model_loaded
        model.eval()
        background = [np.array(timeseries).reshape(1, -1) for timeseries in background]
        background = np.array(background)
        background = torch.tensor(background, dtype=torch.float32)
        instance_processed = torch.tensor(instance_to_be_explained.reshape(1, 1, len(instance_to_be_explained)),
                                          dtype=torch.float32)
    else:
        background = np.array(background)
        instance_processed = np.array(instances_to_be_explained)
        model = lambda x: model_loaded.predict(x)
    # Preprocess your data the same way as classification





    print(f"Processed dataset shape: {background.shape}")
    print(f"Processed instance shape: {instance_processed.shape}")

    # Test that the model works with this preprocessing
    try:
        test_output = model(instance_processed)
        print(f"✓ Model works! Output shape: {test_output.shape}")
        test_all = model(background)
        print(f"✓ Model works all! Output shape: {test_all.shape}")


    except Exception as e:
        print(f"✗ Model error: {e}")
        return None

    # Create explainer and get SHAP values
    explainer = shap.KernelExplainer(model, background)
    print(instance_processed.shape)
    shap_values = explainer.shap_values(instance_processed,gc=False,nsamples=50)

    return shap_values

def get_shap_rank_for_all_time_step(model_name,dataset_name, background, instances_to_be_explained):

    shap_values = get_shap_values(model_name=model_name, dataset_name=dataset_name,background=background,
                                  instances_to_be_explained=instances_to_be_explained)
    print(shap_values.shape)

    # We only want the contributing positive to the predicted class
    to_be_explained_class = get_classes(model_name=model_name, dataset_name=dataset_name,
                                      instances_to_be_explained=instances_to_be_explained)
    print(to_be_explained_class)

    # Extract SHAP values for each class
    c = to_be_explained_class

    shap_class_curr = [shaps if c == 0 else -shaps for shaps,c in zip(shap_values,c)]  # Shape: (24,) - importance for class 0

    all_ts_ranks = []
    for shaps in shap_class_curr:
        diff_vals = set()
        for s in shaps:
            diff_vals.add(s)

        all_counts = sorted(list(diff_vals))
        rank_of_c = {}
        for i,c in enumerate(all_counts):
            rank_of_c[c] = i+1

        rank_of_all_ts = []
        for c in shaps:
            rank_of_all_ts.append(rank_of_c[c])
        all_ts_ranks.append(rank_of_all_ts)
    return all_ts_ranks

def get_most_important_time_step(model_name, dataset_name, time_series_to_be_explained):
    np.random.seed(41)
    dataset = load_dataset(dataset_name=dataset_name, data_type="TEST_normalized")
    num_to_select = min(10, len(dataset))

    num_rows = dataset.shape[0]
    shuffled_indices = np.random.permutation(num_rows)  # random order
    selected_indices = shuffled_indices[:num_to_select]  # pick first k indices
    print("Selected idx:", sorted(selected_indices))

    selected_dataset = dataset[selected_indices]

    shap_values = get_shap_values(model_name=model_name, dataset=selected_dataset, dataset_name=dataset_name,
                                  instance_to_be_explained=time_series_to_be_explained)
    print(shap_values.shape)

    # We only want the contributing positive to the predicted class
    to_be_explained_class = get_class(model_name=model_name, dataset_name=dataset_name,
                                      instance_to_be_explained=time_series_to_be_explained)
    print(to_be_explained_class)

    # Extract SHAP values for each class
    c = to_be_explained_class
    shap_class_curr = shap_values.flatten()
    if c == 0:
        shap_class_curr = -shap_class_curr# Shape: (24,) - importance for class 0

    max_id = np.argmax(shap_class_curr)
    return max_id

if __name__ == "__main__":
    model_name = "miniRocket.pkl"
    dataset_name = "Chinatown"

    np.random.seed(41)
    dataset = load_dataset(dataset_name=dataset_name, data_type="TEST_normalized")
    num_to_select = min(100,len(dataset)-1)

    idx_to_be_explained = 1
    instance_to_be_explained = dataset[idx_to_be_explained]    # Randomly select up to 100 unique elements without replacement

    num_rows = dataset.shape[0]
    shuffled_indices = np.random.permutation(num_rows)  # random order
    shuffled_indices = shuffled_indices[shuffled_indices != idx_to_be_explained]
    selected_indices = shuffled_indices[:num_to_select]  # pick first k indices
    print("Selected idx:",sorted(selected_indices))

    selected_dataset = dataset[selected_indices]

    shap_values = get_shap_values(model_name=model_name, dataset=selected_dataset, dataset_name=dataset_name, instance_to_be_explained=instance_to_be_explained)
    print(shap_values.shape)

    # We only want the contributing positive to the predicted class
    to_be_explained_class = get_classes(model_name=model_name, dataset_name=dataset_name,instance_to_be_explained=instance_to_be_explained)
    print(to_be_explained_class)


    # Extract SHAP values for each class
    c = to_be_explained_class
    shap_values = shap_values.flatten()

    if c == 0:
        # Negative values push towards 0, positive towards 1.
        shap_values = -shap_values

    max_id = np.argmax(shap_values)
    plt.plot(instance_to_be_explained)
    plt.scatter(max_id, instance_to_be_explained[max_id], color="red", marker="o")
    plt.title(f"Most positive for class {c} is index {max_id}")
    plt.show()

    import matplotlib.pyplot as plt

    # Plot SHAP values for the positive class
    plt.figure(figsize=(12, 4))
    plt.plot( shap_values)
    plt.xlabel('Time Step')
    plt.ylabel(f'SHAP Value (Class {c})')
    plt.title('SHAP Values Across Time Steps')
    plt.show()

