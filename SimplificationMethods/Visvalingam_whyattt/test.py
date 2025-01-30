


import numpy as np
from Utils.load_data import load_dataset

from SimplificationMethods.Visvalingam_whyattt.Visvalingam_Whyatt import simplify





def run():
    dataset_name = "Chinatown"
    x_train = load_dataset(dataset_name=dataset_name)
    instance = x_train[0]
    print(simplify(instance, 0.01))

if __name__ == "__main__":
    run()