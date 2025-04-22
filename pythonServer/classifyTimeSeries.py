from Utils.load_models import model_classify
import numpy as np

def _classify(dataset_name, time_series):
    full_path = f"models/{dataset_name}/knn_norm.pkl"
    print("Full path:", full_path)
    pred = model_classify(model_path=full_path, time_series=time_series,num_classes=2)
    return pred


    