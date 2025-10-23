from matplotlib import pyplot as plt
import pandas as pd
from generate_user_survey.configurations import loyalty_value_for_each_dataset, selected_datasets_to_be_in_survey
from Utils.load_data import load_dataset
def visualise_the_selected_loyalty_values(dataset_name, selected_kneepoints):
    simp_algo = "RDP"

    df = pd.read_csv(f"results/{dataset_name}/cnn_alpha_complexity_loyalty.csv")
    df = df[df["Type"]==simp_algo]

    num_segments = df["Num Segments"].tolist()
    loyalty_values = df["Percentage Agreement"].tolist()


    timeseries = load_dataset(dataset_name)
    length = timeseries.shape[1]

    fig, ax = plt.subplots()
    ax.plot([length] + num_segments, [100] +loyalty_values )
    ax.set_xlabel("Number of segments")
    ax.set_ylabel("Percentage Agreement")

    for loyalty,(alpha,num_segs) in selected_kneepoints.items():
        if loyalty.startswith("L"):
            loyalty = float(loyalty[1:])
            ax.scatter(num_segs,loyalty, label=f"Loy:{loyalty}%, Alpha:{alpha}, Seg:{num_segs}")
        elif loyalty.startswith("NoSimp"):
            ax.scatter(num_segs,100, label=f"NoSimp:Loy:{100}%, Alpha:{alpha}, Seg:{num_segs}")



    ax.scatter(length, 100, marker="x", label=f"Full TS:Loy:{100}%, Alpha:N/A, Seg:{length}")
    ax.set_title(f"{dataset_name}")
    ax.legend(loc="lower right")
    fig.savefig(f"generate_user_survey/paper_figs/{dataset_name}_loyalty_values.png")

    fig.show()



if __name__ == "__main__":
    datasets = selected_datasets_to_be_in_survey()
    for dataset_name in datasets:
        selected_loyalty_values = loyalty_value_for_each_dataset(dataset_name)
        selected_kneepoints = selected_loyalty_values
        visualise_the_selected_loyalty_values(dataset_name, selected_kneepoints)
