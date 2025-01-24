import numpy as np
import torch
import joblib

from Utils.conv_model import ConvClassifier

MODELS = {}

def load_pytorch_model(model_path):
    global MODELS
    if model_path not in MODELS:
        pytorch_model = ConvClassifier()
        pytorch_model.load_state_dict(torch.load(model_path, weights_only=False))
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
    return prediction[0]

def model_classify(model_path, time_series):
    if model_path.split("_")[1] == "cnn":
        model = load_pytorch_model(model_path)
        return classify_pytorch_model(model, time_series)
    elif model_path.endswith(".pkl"):
        return classify_sklearn_model(model_path, time_series)
    else:
        raise ValueError("Model path not supported.")
    
def batch_classify_pytorch_model(model, batch_of_timeseries):
    batch_of_timeseries = [np.array(timeseries).reshape(1, -1) for timeseries in batch_of_timeseries]
    batch_of_timeseries = np.array(batch_of_timeseries)
    with torch.no_grad():
        batch_of_timeseries_tensor = torch.tensor(batch_of_timeseries, dtype=torch.float32)
        predictions = model(batch_of_timeseries_tensor).numpy()
    class_pred = [round(prediction[0]) for prediction in predictions]
    return class_pred

def batch_classify_sklearn_model(model_path, batch_of_timeseries):
    model = joblib.load(open(model_path, 'rb'))
    predictions = model.predict(batch_of_timeseries)
    return predictions

def model_batch_classify(model_path, batch_of_timeseries):
    if model_path.split("_")[1] == "cnn":
        model = load_pytorch_model(model_path)
        return batch_classify_pytorch_model(model, batch_of_timeseries)
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