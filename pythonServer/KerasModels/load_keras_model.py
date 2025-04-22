import tensorflow as tf
import numpy as np
from pythonServer.utils.load_csv import load_dataset


def load_keras_model(model_name):
    keras_model = tf.keras.models.load_model(
        f"pythonServer/KerasModels/models/{model_name}")
    return keras_model


def model_classify(model_name, time_series):
    model = load_keras_model(model_name)
    input_shape = model.input_shape[1:]  # Get Shape
    reshaped_input = time_series.reshape(input_shape)
    time_series = np.array([reshaped_input])  # (batchsize=1,(inputshape))
    predictions = model.predict(time_series)
    class_pred = np.argmax(predictions, axis=1)[0]  # Extract batch index 0
    return class_pred


PREDICTIONS = {}


def no_save_batch_classify(model_name,batch_of_timeseries):
    model = load_keras_model(model_name)

    # The Batch should already be correct format
    input_shape = model.input_shape[1:]  # Get Shape
    batch_of_timeseries = [timeseries.reshape(
        input_shape) for timeseries in batch_of_timeseries]
    batch_of_timeseries = np.array(batch_of_timeseries)
    predictions = model.predict(batch_of_timeseries)
    class_pred = [np.argmax(prediction) for prediction in predictions]
    return class_pred

def model_batch_classify(dataset_name, model_name, batch_of_timeseries):
    model_and_data = model_name+dataset_name
    if model_and_data not in PREDICTIONS:
        model = load_keras_model(model_name)

        # The Batch should already be correct format
        input_shape = model.input_shape[1:]  # Get Shape
        batch_of_timeseries = [timeseries.reshape(
            input_shape) for timeseries in batch_of_timeseries]
        batch_of_timeseries = np.array(batch_of_timeseries)
        predictions = model.predict(batch_of_timeseries)
        class_pred = [np.argmax(prediction) for prediction in predictions]
        PREDICTIONS[model_and_data] = class_pred
    return PREDICTIONS[model_and_data]


def model_confidence(dataset, timeseries):
    model = load_keras_model(dataset)
    input_shape = model.input_shape[1:]  # Get Shape
    reshaped_input = timeseries.reshape(input_shape)
    timeseries = np.array([reshaped_input])  # (batchsize=1,(inputshape))
    predictions = model.predict(timeseries)
    confidence = np.max(predictions)
    return confidence
