
import numpy as np
from Utils.load_data import load_dataset

def get_time_series(dataset_name, data_type, index):
    x = load_dataset(dataset_name,data_type=data_type)
    return x[index][1:]

