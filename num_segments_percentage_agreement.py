import os

import pandas as pd
from matplotlib import pyplot as plt


def get_ts_data_from_algo(algos, dataset_name, model_name, metric):
    df = pd.read_csv(f"results/{dataset_name}/{model_name}_alpha_complexity_loyalty.csv")
    algos_dict = {}
    for algo in algos:
        algo_df = df[df["Type"] == algo]
        num_segs = algo_df["Num Segments"].tolist()
        percent_agree = algo_df[(metric)].tolist()
        x_y = sorted(zip(num_segs, percent_agree), key=lambda x: x[0])
        x_es = [x for x, y in x_y]
        y_es = [y for x, y in x_y]
        if x_es[0] == 0:
            x_es = x_es[1:]
            y_es = y_es[1:]
        algos_dict[algo] = (x_es, y_es)
    return algos_dict


def make_and_save_pdf():
    dataset_names = [dataset for dataset in os.listdir("results") if os.path.isdir(f"results/{dataset}")]
    model_name = "miniRocket"
    algos = ["BU", "OS", "RDP", "VW"]
    metrics = ["Percentage Agreement", "Kappa Loyalty"]
    for metric in metrics:
        for dataset_name in dataset_names:
            algos_dict = get_ts_data_from_algo(dataset_name=dataset_name, algos=algos, model_name="miniRocket",
                                               metric=metric)

            for algo in algos_dict.keys():
                x_es, y_es = algos_dict[algo]
                plt.plot(x_es, y_es, label=algo)
            plt.legend(loc="lower right")
            plt.xlabel("Number of Segments")
            plt.ylabel(metric)
            plt.title(f"Num. Segments vs. {metric} {dataset_name}")
            plt.savefig(f"results/{dataset_name}/{model_name}_numSegments_{'_'.join(metric.split())}.png")

            plt.show()
            plt.clf()


if __name__ == "__main__":
    make_and_save_pdf()
