from typing import Tuple, Dict

import numpy as np
import pandas as pd


def find_alpha_given_loyalty(model_name: str, algo: str, dataset_name: str, threshold: float, measure: str):
    df = pd.read_csv(f"results/{dataset_name}/{model_name}_alpha_complexity_loyalty.csv")
    df = df[df[measure] == threshold]
    df = df[df["Type"] == algo]
    if df.empty:
        return None
    min_row_idx = np.argmin(df["Num Segments"])
    min_row = df.iloc[min_row_idx]
    alpha = min_row["Alpha"]
    num_segments = min_row["Num Segments"]
    return alpha, num_segments


def config_loyalty(dataset_name) -> Dict[str, Tuple[float, float]]:
    """

    """
    config_details = {
        'ECG200': {
            'L82': (0.01, 3.07),
            'L85': (0.02, 3.84),
            'L91': (0.03, 4.79),
            'L92': (0.05, 6.11),
            'L94': (0.06, 6.66),
            'L95': (0.1, 10.41),
            'L97': (0.07, 7.71),
            'L98': (0.16, 14.31),
            'L100': (0.23, 20.36),
        },
        'Chinatown': {
            'L72': (0.01, 1.0),
            'L87': (0.04, 2.17),
            'L89': (0.08, 2.91),
            'L91': (0.07, 2.74),
            'L92': (0.1, 3.2),
            'L95': (0.05, 2.41),
            'L96': (0.16, 4.31),
            'L98': (0.17, 4.6),
            'L99': (0.21, 5.47),
            'L100': (0.45, 8.33),
        },
        'SonyAIBORobotSurface1': {
            'L71': (0.04, 6.88),
            'L77': (0.05, 8.48),
            'L83': (0.06, 9.9),
            'L89': (0.07, 10.98),
            'L90': (0.08, 11.62),
            'L92': (0.12, 14.48),
            'L93': (0.13, 15.08),
            'L96': (0.14, 15.87),
            'L97': (0.19, 19.55),
            'L98': (0.25, 24.11),
            'L99': (0.26, 24.72),
            'L100': (0.28, 26.35),
        },
        'BME': {
            'L73': (0.01, 5.44),
            'L96': (0.02, 6.61),
            'L99': (0.17, 13.68),
            'L100': (0.03, 7.59),
        },
        'UMD': {
            'L91': (0.02, 7.06),
            'L95': (0.03, 7.92),
            'L96': (0.06, 9.44),
            'L97': (0.13, 11.71),
            'L98': (0.15, 12.95),
            'L99': (0.16, 13.6),
            'L100': (0.4, 24.51),
        }, }

    return config_details[dataset_name]


if __name__ == "__main__":
    model_name = "miniRocket"
    thresholds = [i for i in range(70, 101)]
    dataset_names = ["ECG200", "Chinatown", "SonyAIBORobotSurface1", "BME", "UMD"]
    metric = "Percentage Agreement"
    algo = "OS"
    for dataset_name in dataset_names:
        print("'" + dataset_name + "':{")
        for threshold in thresholds:
            # Might return None
            alpha_num_segments = find_alpha_given_loyalty(model_name, algo, dataset_name, threshold, metric)
            if alpha_num_segments:
                alpha, num_segments = alpha_num_segments
                print(f"\t'L{threshold}': ({round(alpha, 2)},{round(num_segments, 2)}),")
        print("},")
