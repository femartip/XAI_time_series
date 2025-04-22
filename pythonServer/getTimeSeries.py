
import numpy as np
from pythonServer.utils.load_csv import load_dataset


def get_time_series(data_set_name, index):
    x = load_dataset(data_set_name)
    return x[index]
