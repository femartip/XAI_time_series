from Utils.load_data import normalize_data

if __name__ == "__main__":
    datasets = ["Chinatown", "ECG200", "ItalyPowerDemand"]
    datatypes = ["TRAIN", "TEST", "VALIDATION"]
    for dataset in datasets:
        for datatype in datatypes:
            normalize_data(dataset, datatype)