import pandas as pd

def selected_datasets_to_be_in_survey():
    return ["Chinatown", "ECG200", "SonyAIBORobotSurface1","ProximalPhalanxOutlineCorrect"]


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
        "ProximalPhalanxOutlineCorrect": {
            "L81": (0.13, 8.99),
            "L96": (0.09, 9.78),
            "L100": (0.05, 13.3),
            "NoSimp": (0.0, 64.26)
        }
    }

    return dataset_knees[dataset_name]



