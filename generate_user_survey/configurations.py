import numpy as np
import pandas as pd


def selected_datasets_to_be_in_survey():
    return ["Chinatown", "ECG200", "UMD", "ProximalPhalanxOutlineAgeGroup"]


def get_no_simp_data(dataset_name, model_name):
    df = pd.read_csv(f"results/{dataset_name}/{model_name}_alpha_complexity_loyalty.csv")
    df = df[(df["Type"] == "RDP") & (df["Alpha"] == 0)]  # Maintains everything

    min_segments_idx = np.argmin(df["Num Segments"])

    top_row = df.iloc[min_segments_idx]
    num_seg = top_row["Num Segments"]
    loyalty = top_row["Percentage Agreement"]
    alpha = top_row["Alpha"]
    assert loyalty == 100

    return alpha, num_seg


def _print_no_simp_info():
    # Print the no simp for each dataset
    dataset_names = selected_datasets_to_be_in_survey()
    model_name = "miniRocket"
    for dataset_name in dataset_names:
        alpha, num_seg = get_no_simp_data(dataset_name=dataset_name, model_name=model_name)
        print(dataset_name, f"'NoSimp': ({alpha}, {num_seg}),")


def loyalty_value_for_each_dataset(dataset_name):
    """
    Format:
      dataset_name:
            loyalty:
                (alpha, num_segments)

    """
    dataset_knees = {
        'ECG200': {
            'L82': (0.01, 3.07),
            'L97': (0.07, 7.71),
            'L100': (0.23, 20.36),
            'NoSimp': (0.0, 95.0),
        },
        'Chinatown': {
            'L72': (0.01, 1.0),
            'L95': (0.05, 2.41),
            'L100': (0.45, 8.33),
            'NoSimp': (0.0, 22.97),
        },
        'SonyAIBORobotSurface1': {
            'L90': (0.08, 11.62),
            'L96': (0.14, 15.87),
            'L100': (0.28, 26.35),
            'NoSimp': (0.0, 60.67),
        },
        'BME': {
            'L73': (0.01, 5.44),
            'L96': (0.02, 6.61),
            'L100': (0.03, 7.59),
            'NoSimp': (0.0, 124.35),
        },
        'UMD': {
            'L95': (0.03, 7.92),
            'L99': (0.16, 13.6),
            'L100': (0.4, 24.51),
            'NoSimp': (0.0, 146.75),
        },
        'ProximalPhalanxOutlineAgeGroup':{
        'L91': (0.08,9.63),
        'L95': (0.27,17.79),
        'L100': (0.73,39.63),
        'NoSimp': (0.0, 65.23),
    },
    }

    return dataset_knees[dataset_name]


def _config_to_dataset_and_loyalty_print():
    """
    Use this to make a new hardcoded config setup
    """
    datasets = selected_datasets_to_be_in_survey()

    configs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    config_details = {}
    for i, dataset in enumerate(datasets):
        loyalty_value = loyalty_value_for_each_dataset(dataset)
        for j, loyalty_level in enumerate(loyalty_value.keys()):
            config_details[configs[(i * 4) + j]] = (dataset, loyalty_level)

    import json
    print(json.dumps(config_details, indent=4))


def get_dataset_and_loyalty_from_config(config):
    config_to_dataset_and_loyalty = {
    "A": [
        "Chinatown",
        "L72"
    ],
    "B": [
        "Chinatown",
        "L95"
    ],
    "C": [
        "Chinatown",
        "L100"
    ],
    "D": [
        "Chinatown",
        "NoSimp"
    ],
    "E": [
        "ECG200",
        "L82"
    ],
    "F": [
        "ECG200",
        "L97"
    ],
    "G": [
        "ECG200",
        "L100"
    ],
    "H": [
        "ECG200",
        "NoSimp"
    ],
    "I": [
        "UMD",
        "L95"
    ],
    "J": [
        "UMD",
        "L99"
    ],
    "K": [
        "UMD",
        "L100"
    ],
    "L": [
        "UMD",
        "NoSimp"
    ],
    "M": [
        "ProximalPhalanxOutlineAgeGroup",
        "L91"
    ],
    "N": [
        "ProximalPhalanxOutlineAgeGroup",
        "L95"
    ],
    "O": [
        "ProximalPhalanxOutlineAgeGroup",
        "L100"
    ],
    "P": [
        "ProximalPhalanxOutlineAgeGroup",
        "NoSimp"
    ]
    }
    return config_to_dataset_and_loyalty[config]


def get_config_of_group(group):
    groups_to_configs = {
        "G1": ["A", "F", "K", "P"],
        "G2": ["D", "G", "J", "M"],
        "G3": ["B", "H", "I", "O"],
        "G4": ["C", "E", "L", "N"]
    }
    return groups_to_configs[group]


if __name__ == "__main__":
    _config_to_dataset_and_loyalty_print()
    # _print_no_simp_info()
    pass
