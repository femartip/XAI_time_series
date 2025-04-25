import numpy as np
import pandas as pd
from sklearn import preprocessing

"""
This code has been extracted from: 
https://github.com/BrigtHaavardstun/kSimplification
"""


def read_numpy(dataset_name: str) -> np.ndarray:
    """
    Parse the data from TSV file into a Dataframe, and transform it into a numpy array.
    :param dataset_name:
    :return:
    """
    folder = "./data/" + dataset_name.split("_")[0] + "/"
    file_location = folder + dataset_name
    array_2d = np.load(file_location)

    return array_2d


def zero_indexing_labels(current_labels: np.ndarray, dataset: str) -> np.ndarray:
    """
    Encodes the labels as zero index.
    For instance: labels: e.g. 1,2,3,4,... -> go to -> labels: 0,1,2,3,...

    :param current_labels:
    :param dataset:
    :return:
    """
    training_labels = load_dataset_org_labels(dataset, data_type="TRAIN")
    test_labels = load_dataset_org_labels(dataset, data_type="TEST")
    validation_labels = load_dataset_org_labels(dataset, data_type="VALIDATION")
    le = preprocessing.LabelEncoder()
    le.fit(np.concatenate([training_labels, test_labels, validation_labels], axis=0))
    transformed_labels = le.transform(current_labels)
    return transformed_labels


def load_data_set_full(dataset_name: str, data_type: str = "TRAIN") -> np.ndarray:
    array_2d = read_numpy(dataset_name + "_" + data_type + ".npy")
    return array_2d


def load_dataset(dataset_name: str, data_type: str = "TRAIN") -> np.ndarray:
    """
    Load all time series in {train/test} dataset.
    :param data_type:
    :param dataset_name:
    :return: 2D numpy array
    """
    array_2d = load_data_set_full(dataset_name=dataset_name, data_type=data_type)
    
    # Remove the first column (index 0) along axis 1 (columns)
    data = np.delete(array_2d, 0, axis=1)
    return data


def load_dataset_org_labels(dataset_name: str, data_type: str = "TRAIN") -> np.ndarray:
    """
    Load the labels from the dataset
    :param data_type:
    :param dataset_name:
    :return:
    """
    array_2d = load_data_set_full(dataset_name, data_type=data_type)

    # Keep only the first column (index 0)
    array_2d = array_2d[:, 0]
    return array_2d


def load_dataset_labels(dataset_name, data_type: str = "TRAIN") -> np.ndarray:
    """
    Load the labels AND onehot encode them.
    :param data_type:
    :param dataset_name:
    :return:
    """
    labels_current = load_dataset_org_labels(dataset_name, data_type=data_type)
    zero_indexed = zero_indexing_labels(labels_current, dataset_name)
    return zero_indexed

def load_raw_dataset_labels(dataset_name, data_type: str = "TRAIN") -> np.ndarray:
    """
    Load the labels AND onehot encode them.
    :param data_type:
    :param dataset_name:
    :return:
    """
    labels_current = load_dataset_org_labels(dataset_name, data_type=data_type)
    return labels_current

def get_time_series(dataset_name: str, data_type:str, instance_nr: int):
    all_time_series = load_dataset(dataset_name, data_type=data_type)
    return all_time_series[instance_nr]

def test():
    data = load_dataset("Chinatown", data_type="VALIDATION")
    print(data.shape)
    print(data)

def normalize_data(dataset_name: str, data_type: str = "TRAIN"):
    dataset = load_dataset(dataset_name, data_type)
    
    max_over_all = np.max(dataset)
    min_over_all = np.min(dataset)
    
    dataset = (dataset - min_over_all) / (max_over_all - min_over_all + 1e-8)
    file_path_name = "data/" + dataset_name + "/" +dataset_name +"_" + data_type + "_normalized.npy"
    labels = load_raw_dataset_labels(dataset_name, data_type)
    dataset = np.hstack((labels.reshape(-1, 1), dataset))
    assert not np.isnan(dataset).any(), f"NaN values in the dataset {dataset}."
    np.save(file_path_name, dataset)

if __name__ == "__main__":
    test()
