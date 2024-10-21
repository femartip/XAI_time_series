import numpy as np
from matplotlib import pyplot as plt
import argparse

from Utils.load_data import load_dataset

from ORSalgorithm.simplify.DPcustomAlgoKSmallest import solve_and_find_points
from ORSalgorithm.Utils.data import get_min_and_max, dataset_sensitive_c
from ORSalgorithm.Utils.line import interpolate_points_to_line

def generate_approximation_ts_for_all_in_dataset(dataset_name, my_k, alpha):
    all_time_series = load_dataset(dataset_name, data_type="TEST")

    min_y, max_y = get_min_and_max(all_time_series)
    distance_weight = max_y - min_y

    my_c = dataset_sensitive_c(dataset=dataset_name, distance_weight=distance_weight) 
    for ts_nr in range(len(all_time_series)):
        print("TS number:", ts_nr)
        ts = all_time_series[ts_nr]
        print(f"TS: {ts}")

        x_values = [i for i in range(len(ts))]

        all_selected_points, all_ys = solve_and_find_points(x_values, ts, c=my_c, K=my_k, saveImg=False, distance_weight=distance_weight, alpha=alpha)
        all_interpolations = []
        
        for i, (selected_points, ys) in enumerate(zip(all_selected_points, all_ys)):
            inter_ts = interpolate_points_to_line(ts_length=len(ts), x_selected=selected_points, y_selected=ys)
            all_interpolations.append(inter_ts)

        for i, (inter_ts, selected_points, ys) in enumerate(zip(all_interpolations, all_selected_points, all_ys)):
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            
            axs[0][0].plot(x_values, ts, '--x', color='black')
            axs[0][0].set_title('Original Time Series')
            
            axs[0][1].plot(x_values, inter_ts, '--o', color='blue')
            axs[0][1].set_title('Interpolated Time Series')
            
            axs[1][0].plot(selected_points, ys, '--D', color='red')
            axs[1][0].set_title('Selected Points')
            
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(1)
            plt.close(fig)
            

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_name", type=str, default="Chinatown", help="Name of the dataset, can be either Chinatown, ECG200 or ItalyEnergy")
    args.add_argument("--k", type=int, default=1, help="Number of k best solutions")
    args.add_argument("--alpha", type=float, default=0.2)
    args = args.parse_args()
    
    generate_approximation_ts_for_all_in_dataset(args.dataset_name, args.k, args.alpha)
