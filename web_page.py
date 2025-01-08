import flask
from flask import Flask, render_template, send_file, request
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from matplotlib import pyplot as plt
import argparse
import logging

from Utils.load_data import load_dataset  
from ORSalgorithm.ORSalgorithm import ORSalgorithm  

app = Flask(__name__)

def generate_time_series_plot(original_ts, simplified_ts, title="Time Series Plot"):
    """Generates a time-series plot with original and simplified data and returns it as a base64 encoded image."""
    plt.figure()
    plt.plot(original_ts, label="Original")
    plt.plot(simplified_ts, label="Simplified")
    plt.title(title)
    plt.legend()

    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    plt.close()

    return img_str

def generate_prototypes_for_best_alpha(dataset_name, model_path, my_k):
    all_time_series = load_dataset(dataset_name, data_type="TRAIN")
    best_alphas = []
    best_confidences = []
    best_simplifications = []
    
    plot_data = []

    for i, ts in enumerate(all_time_series):
        best_alpha = None
        best_confidence = -np.inf
        best_simplification = None

        for a in np.arange(0.1, 1.1, 0.1):
            simplification, confidence, chosen_class = ORSalgorithm(np.array([ts]), model_path, k=my_k, alpha=a)
            avg_confidence = np.mean(confidence)
            if avg_confidence > best_confidence:
                best_confidence = avg_confidence
                best_alpha = a
                best_simplification = simplification

        best_alphas.append(best_alpha)
        best_confidences.append(best_confidence)
        best_simplifications.append(best_simplification)
        
        title = f"Time series {i} with best alpha={best_alpha:.2f} and confidence={best_confidence:.2f}"
        img_str = generate_time_series_plot(ts, best_simplification[0], title)
        plot_data.append(img_str)

    return plot_data

def generate_prototypes(dataset_name, model_path, k, alpha):
    all_time_series = load_dataset(dataset_name, data_type="TRAIN")
    simplification, confidence, chosen_class = ORSalgorithm(all_time_series, model_path, k=k, alpha=alpha)

    plot_data = []
    for i in range(len(all_time_series)):
        title = f"Time series {i} with alpha={alpha:.2f} and confidence={confidence[i]:.2f}"
        img_str = generate_time_series_plot(all_time_series[i], simplification[i], title)
        plot_data.append(img_str)
        
    return plot_data


@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/get_plot', methods=['POST'])
def get_plot():
    """Generates and returns the requested time series plot."""
    dataset_name = request.form.get('dataset_name', 'Chinatown')
    model_path = request.form.get('model_path', 'models/chinatown_base.pth')
    k = int(request.form.get('k', 1))
    alpha = float(request.form.get('alpha', 0.02))
    plot_type = request.form.get('plot_type', 'prototype')  # Default to 'prototype'

    if plot_type == 'prototype':
        plot_data = generate_prototypes(dataset_name, model_path, k, alpha)
    elif plot_type == 'best_prototype':
        plot_data = generate_prototypes_for_best_alpha(dataset_name, model_path, k)
    else:
        return "Invalid plot type", 400
        
    return render_template('plot.html', plot_data=plot_data)


if __name__ == '__main__':

    app.run(debug=True)