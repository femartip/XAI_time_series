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
            'L72': (0.27, 4.07),
            'L74': (0.26, 4.16),
            'L77': (0.25, 4.37),
            'L81': (0.23, 4.64),
            'L84': (0.22, 4.8),
            'L87': (0.21, 4.87),
            'L88': (0.18, 5.3),
            'L90': (0.19, 5.16),
            'L91': (0.16, 5.74),
            'L92': (0.15, 6.09),
            'L94': (0.12, 7.0),
            'L95': (0.14, 6.33),
            'L98': (0.07, 12.67),
            'L100': (0.02, 40.37),
        },
        'Chinatown': {
            'L70': (0.3, 2.9),
            'L71': (0.29, 2.95),
            'L72': (0.31, 2.82),
            'L74': (0.53, 1.33),
            'L75': (0.5, 1.5),
            'L76': (0.55, 1.2),
            'L77': (0.62, 1.03),
            'L78': (0.66, 1.0),
            'L79': (0.35, 2.55),
            'L80': (0.38, 2.32),
            'L81': (0.45, 1.93),
            'L83': (0.42, 2.11),
            'L84': (0.4, 2.25),
            'L86': (0.22, 3.86),
            'L88': (0.21, 3.95),
            'L90': (0.2, 4.09),
            'L92': (0.19, 4.26),
            'L93': (0.17, 4.41),
            'L94': (0.16, 4.53),
            'L96': (0.15, 4.7),
            'L97': (0.1, 5.81),
            'L98': (0.05, 9.3),
            'L99': (0.04, 10.53),
            'L100': (0.03, 12.33),
        },
        'SonyAIBORobotSurface1': {
            'L73': (0.22, 7.51),
            'L74': (0.21, 7.83),
            'L76': (0.2, 8.3),
            'L79': (0.19, 8.72),
            'L83': (0.17, 9.79),
            'L87': (0.16, 10.3),
            'L88': (0.15, 10.82),
            'L89': (0.14, 11.22),
            'L91': (0.13, 11.93),
            'L92': (0.1, 13.64),
            'L94': (0.09, 14.69),
            'L95': (0.08, 15.97),
            'L96': (0.07, 17.72),
            'L98': (0.02, 38.95),
            'L99': (0.05, 22.63),
            'L100': (0.03, 31.33),
        },
        'PhalangesOutlinesCorrect': {
            'L71': (0.16, 7.1),
            'L73': (0.12, 8.22),
            'L74': (0.15, 7.45),
            'L77': (0.11, 8.49),
            'L78': (0.09, 9.13),
            'L79': (0.1, 8.88),
            'L80': (0.05, 11.72),
            'L81': (0.08, 9.51),
            'L82': (0.04, 13.35),
            'L85': (0.06, 10.59),
            'L86': (0.07, 9.92),
            'L90': (0.03, 16.19),
            'L97': (0.02, 20.72),
            'L100': (0.01, 30.68),
        },
    }

    return config_details[dataset_name]


if __name__ == "__main__":
    model_name = "miniRocket"
    thresholds = [i for i in range(70, 101)]
    dataset_names = ["ECG200", "Chinatown", "SonyAIBORobotSurface1", "PhalangesOutlinesCorrect"]
    metric = "Percentage Agreement"
    algo = "RDP"
    for dataset_name in dataset_names:
        print("'" + dataset_name + "':{")
        for threshold in thresholds:
            # Might return None
            alpha_num_segments = find_alpha_given_loyalty(model_name, algo, dataset_name, threshold, metric)
            if alpha_num_segments:
                alpha, num_segments = alpha_num_segments
                print(f"\t'L{threshold}': ({round(alpha, 2)},{round(num_segments, 2)}),")
        print("},")
