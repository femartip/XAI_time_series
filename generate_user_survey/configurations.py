import pandas as pd

def selected_datasets_to_be_in_survey():
    return ["Chinatown", "ECG200", "SonyAIBORobotSurface1","PhalangesOutlinesCorrect"]


def get_no_simp_data(dataset_name, simp_algo="RDP"):
    df = pd.read_csv(f"results/{dataset_name}/cnn_alpha_complexity_loyalty.csv")
    df = df[df["Type"] == simp_algo]

    max_idx = df["Percentage Agreement"].idxmax()
    top_row = df.loc[max_idx]

    num_seg = top_row["Num Segments"]
    loyalty = top_row["Percentage Agreement"]
    alpha = top_row["Alpha"]

    return alpha,num_seg


def loyalty_value_for_each_dataset(dataset_name):
    """
    Format:
      dataset_name:
            loyalty:
                (alpha, num_segments)

    """
    dataset_knees = {
        "ECG200":{
            "L85": (0.97, 1.44),
            "L91":(0.18, 6.4),
            "L100":(0.03, 30.07),
            "NoSimp": (0.0, 96.0)
        },
        "Chinatown": {
            "L78": (0.66, 1.0),
            "L93": (0.04, 10.53),
            "L100": (0.01, 17.5),
            "NoSimp": (0.0, 22.97)
        },
        "SonyAIBORobotSurface1": {
            "L88": (0.09, 15.74),
            "L93": (0.05, 23.7),
            "L100": (0.01, 52.98),
            "NoSimp": (0.0, 61.67)
        },
        "PhalangesOutlinesCorrect": {
            "L96": (0.15, 8.45),
            "L99": (0.1, 9.87),
            "L100": (0.04, 14.37),
            "NoSimp": (0.0, 70.15)
        }
    }

    return dataset_knees[dataset_name]

def _config_to_dataset_and_loyalty_print():
    """
    Use this to make a new hardcoded config setup
    """
    datasets = selected_datasets_to_be_in_survey()

    configs = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]
    config_details = {}
    for i,dataset in enumerate(datasets):
        loyalty_value = loyalty_value_for_each_dataset(dataset)
        for j,loyalty_level in enumerate(loyalty_value.keys()):
            config_details[configs[(i*4)+j]] = (dataset,loyalty_level)

    import json
    print(json.dumps(config_details, indent=4))

def get_dataset_and_loyalty_from_config(config):

    config_to_dataset_and_loyalty = {
        "A": [
            "Chinatown",
            "L78"
        ],
        "B": [
            "Chinatown",
            "L93"
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
            "L85"
        ],
        "F": [
            "ECG200",
            "L91"
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
            "SonyAIBORobotSurface1",
            "L88"
        ],
        "J": [
            "SonyAIBORobotSurface1",
            "L93"
        ],
        "K": [
            "SonyAIBORobotSurface1",
            "L100"
        ],
        "L": [
            "SonyAIBORobotSurface1",
            "NoSimp"
        ],
        "M": [
            "PhalangesOutlinesCorrect",
            "L96"
        ],
        "N": [
            "PhalangesOutlinesCorrect",
            "L99"
        ],
        "O": [
            "PhalangesOutlinesCorrect",
            "L100"
        ],
        "P": [
            "PhalangesOutlinesCorrect",
            "NoSimp"
        ]
    }
    return config_to_dataset_and_loyalty[config]

def get_config_of_group(group):
    groups_to_configs = {
        "G1": ["A","F","K","P"],
        "G2": ["D", "G", "J","M"],
        "G3": ["B","H","I","O"],
        "G4": ["C","E","L","N"]
    }
    return groups_to_configs[group]

if __name__ == "__main__":
    print(get_no_simp_data(dataset_name="PhalangesOutlinesCorrect", simp_algo="RDP"))