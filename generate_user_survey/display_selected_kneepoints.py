from matplotlib import pyplot as plt
import pandas as pd
from generate_usersurvey.generate_full_user_survey import loyalty_value_for_each_dataset
def visualise_the_selected_loyalty_values(dataset_name, selected_kneepoints):
    simp_algo = "RDP"

    df = pd.read_csv(f"results/{dataset_name}/cnn_alpha_complexity_loyalty.csv")
    df = df[df["Type"]==simp_algo]

    num_segments = df["Num Segments"]
    loyalty_values = df["Percentage Agreement"]

    fig, ax = plt.subplots()
    ax.plot(num_segments, loyalty_values)
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Percentage Agreement")

    for loyalty,(alpha,num_segs) in selected_kneepoints.items():
        ax.scatter(num_segs,loyalty)
    ax.set_title(f"{dataset_name}")
    fig.show()


if __name__ == "__main__":
    dataset_name = "Chinatown"
    selected_loyalty_values = loyalty_value_for_each_dataset(dataset_name)
    selected_kneepoints = selected_loyalty_values
    visualise_the_selected_loyalty_values(dataset_name, selected_kneepoints)
