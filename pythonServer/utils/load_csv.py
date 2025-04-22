import numpy as np
import pandas as pd
def load_dataset(dataset_name):
    folder = "pythonServer/utils/csvData/"
    file_location = folder + dataset_name
    array_2d = pd.read_csv(file_location, header=None)
    array_2d = np.array(array_2d)

    return array_2d


def run():
    dataset = load_dataset("Chinatown.csv")
    print("hello")


if __name__ == "__main__":
    run()



