import os

import joblib
import numpy as np
import torch

from Utils.conv_model import ConvClassifier

MODELS = {}
def load_model(model_path, num_classes):
    if model_path not in MODELS:
        if model_path.endswith('.pth'):
            MODELS[model_path] = load_pytorch_model(model_path,num_classes)
        else:
            MODELS[model_path] = load_sklearn_model(model_path)
    return MODELS[model_path]

def load_sklearn_model(model_path):
    return joblib.load(model_path)

def load_pytorch_model(model_path, num_classes: int):
    global MODELS
    if model_path not in MODELS:
        num_classes = 1 if num_classes == 2 else num_classes
        pytorch_model = ConvClassifier(num_classes=num_classes)
        pytorch_model.load_state_dict(torch.load(model_path, weights_only=False,map_location=torch.device('cpu')))
        pytorch_model.eval()  # Set the model to evaluation mode
        MODELS[model_path] = pytorch_model
    return MODELS[model_path]

def classify_pytorch_model(model, time_series):
    time_series = np.array(time_series).reshape(1, -1) # Reshape to (1, input_shape)
    time_series = np.array([time_series])  # (batchsize=1,(inputshape))
    with torch.no_grad():
        time_series_tensor = torch.tensor(time_series, dtype=torch.float32)
        predictions = model(time_series_tensor).numpy()
    class_pred = round(predictions[0][0])  # Extract batch index 0
    return class_pred

def classify_sklearn_model(model_path, time_series):
    model = joblib.load(open(model_path, 'rb'))
    prediction = model.predict(time_series)
    #if len(np.unique(prediction)) > 2:
    #    return 1 if prediction[0] > 0.5 else 0
    return prediction[0]

def model_classify(model_path: str, time_series: list[float], num_classes: int) -> int:
    assert os.path.exists(model_path), f"Model path {model_path} does not exist"

    if model_path.split("_")[1] == "cnn":
        model = load_pytorch_model(model_path, num_classes)
        return classify_pytorch_model(model, time_series)
    elif model_path.endswith(".pkl"):
        return classify_sklearn_model(model_path, time_series)
    else:
        raise ValueError("Model path not supported.")
    
def batch_classify_pytorch_model(model, batch_of_timeseries, num_classes: int):
    batch_of_timeseries = [np.array(timeseries).reshape(1, -1) for timeseries in batch_of_timeseries]
    batch_of_timeseries = np.array(batch_of_timeseries)
    with torch.no_grad():
        batch_of_timeseries_tensor = torch.tensor(batch_of_timeseries, dtype=torch.float32)
        predictions = model(batch_of_timeseries_tensor)

        if num_classes == 2:
            predictions = torch.sigmoid(input=predictions).numpy()
            class_pred = [1 if pred > 0.5 else 0 for pred in predictions]
        else:
            class_pred = torch.softmax(input=predictions, dim=1).numpy()
            class_pred = np.argmax(class_pred, axis=1).tolist()

    return class_pred


def batch_classify_sklearn_model(model_path, batch_of_timeseries):
    model = joblib.load(open(model_path, 'rb'))

    batch_of_timeseries = np.array(batch_of_timeseries)
    predictions = model.predict(batch_of_timeseries)
    # if len(np.unique(predictions)) > 2:
    #    predictions = [1 if pred > 0.5 else 0 for pred in predictions]
    return predictions


def model_batch_classify(model_path: str, batch_of_timeseries: list[list[float]], num_classes: int = None) -> list[int]:
    assert os.path.exists(model_path), f"Model path {model_path} does not exist"

    if model_path.endswith(".pth"):
        assert num_classes is not None, f"num_classes is None for {model_path}"
        model = load_pytorch_model(model_path, num_classes)
        return batch_classify_pytorch_model(model, batch_of_timeseries, num_classes)
    elif model_path.endswith(".pkl"):
        return batch_classify_sklearn_model(model_path, batch_of_timeseries)
    else:
        raise ValueError("Model path not supported.")


"""
def model_confidence(dataset, timeseries):
    model = load_pytorch_model(dataset)
    reshaped_input = timeseries.reshape(1, -1)  # Reshape to (1, input_shape)
    timeseries = np.array([reshaped_input])  # (batchsize=1,(inputshape))
    with torch.no_grad():
        timeseries_tensor = torch.tensor(timeseries, dtype=torch.float32)
        predictions = model(timeseries_tensor).numpy()
    confidence = np.max(predictions)
    return confidence


def batch_confidence(model_path, batch_of_timeseries):
    model = load_pytorch_model(model_path)
    # The Batch should already be correct format
    batch_of_timeseries = [np.array(timeseries).reshape(1, -1) for timeseries in batch_of_timeseries]
    batch_of_timeseries = np.array(batch_of_timeseries)
    with torch.no_grad():
        batch_of_timeseries_tensor = torch.tensor(batch_of_timeseries, dtype=torch.float32)
        predictions = model(batch_of_timeseries_tensor).numpy()
    confidence_pred = np.array([np.max(prediction) for prediction in predictions])
    return confidence_pred
"""
