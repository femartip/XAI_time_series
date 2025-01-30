from SimplificationMethods.SWAB.approximation import Approximation
from SimplificationMethods.SWAB.segmentation import Segmentation
from Utils.load_data import load_dataset


def simplify(instance):
    approx = Approximation()




def run():
    dataset_name = "Chinatown"
    x_train = load_dataset(dataset_name=dataset_name)
    instance = x_train[0]

if __name__ == "__main__":
    run()