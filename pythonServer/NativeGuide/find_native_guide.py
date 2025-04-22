from tslearn.neighbors import KNeighborsTimeSeries
from pythonServer.KerasModels.load_keras_model import model_classify, model_batch_classify
from pythonServer.utils.load_csv import load_dataset as load_dataset_csv
import numpy as np
import pandas as pd


def native_guide_retrieval(query, predicted_label, distance, n_neighbors, dataset, model):
    x = load_dataset_csv(dataset)
    y_pred = model_batch_classify(dataset, model, x)

    df = pd.DataFrame(y_pred, columns=['label'])
    df.index.name = 'index'

    ts_length = x.shape[1]

    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric=distance)

    knn.fit(x[list(df[df['label'] != predicted_label].index.values)])

    dist, ind = knn.kneighbors(query.reshape(
        1, ts_length), return_distance=True)
    print(df[df['label'] != predicted_label].index[ind[0][:]])
    return df[df['label'] != predicted_label].index[ind[0][:]]


def find_native_cf(instance, dataset_name, model_name):
    # Get label
    pred_label = model_classify(model_name, instance)

    # Get NUN of instance
    nun_idx = native_guide_retrieval(
        instance, pred_label, 'euclidean', 1, dataset_name, model_name)[0]
    x = load_dataset_csv(dataset_name)

    nun_cf = x[nun_idx]
    # nun_cf = [val[0] for val in nun_cf]
    nun_cf = np.asarray(nun_cf)
    return nun_cf
