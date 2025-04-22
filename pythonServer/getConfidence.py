# from Blackbox_classifier_FCN.LITE.predict import predict_lite
from pythonServer.KerasModels.load_keras_model import model_confidence
from pythonServer.utils.common import get_domains
import numpy as np
import tensorflow as tf
from keras import backend as K

import threading

# Thread-local storage for our TensorFlow model
# mode_ITALY = load_model("ItalyPowerDemand")


def get_confidence(time_series: np.ndarray, data_set: str) -> float:
    confidence = None
    try:
        confidence = model_confidence(data_set, time_series)

    except Exception as e:
        print(e)

    return confidence
