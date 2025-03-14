import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score
from typing import List
import pandas as pd
from kneed import KneeLocator

from Utils.dataTypes import SegmentedTS 

def score_simplicity(approximation: SegmentedTS) -> float:
        if approximation.num_real_segments is None:
            simplicity = (len(approximation.x_pivots) - 1)  * (1 / (len(approximation.line_version) - 1))
        else:
            simplicity = approximation.num_real_segments  * (1 / (len(approximation.line_version) - 1))
            
        return simplicity

def calculate_mean_loyalty(pred_class_original:List[int], pred_class_simplified:List[int])->float:
    """
    Calculate Mean score to measure agreement between original and simplified classifications.
    """
    loyalty = np.mean(np.equal(pred_class_original, pred_class_simplified))
    return loyalty

def calculate_kappa_loyalty(pred_class_original:List[int], pred_class_simplified:List[int])->float:
    """
    Calculate Cohen's Kappa score to measure agreement between original and simplified classifications.
    """
    #https://github.com/scikit-learn/scikit-learn/issues/9624 
    if len(set(pred_class_original).union(set(pred_class_simplified))) == 1:
        kappa_loyalty = 1.0
    else:
        kappa_loyalty = cohen_kappa_score(pred_class_original, pred_class_simplified, labels=[0,1])
    
    return kappa_loyalty

def calculate_complexity(batch_simplified_ts: List[SegmentedTS])->float:
    """
    Calculate complexity of simplified time series as mean number of segments.
    """
    complexity = np.mean([score_simplicity(ts) for ts in batch_simplified_ts])
    return complexity

def auc(df: pd.DataFrame, metric:str="Kappa Loyalty", show_fig:bool=False) -> tuple[dict[str, float], dict[str,tuple[List, List]]]:
    """
    Calculate the Area Under the Curve of the Complexity vs Loyalty curve for each simplification algorithm.
    This is used to compare the performance of the different simplification algorithms.
    """
    assert metric != "Kappa Loyalty" or metric != "Mean Loyalty", "Metric must be either Kappa Loyalty or Mean Loyalty"

    algorithms = df["Type"].unique()
    auc = {}
    filtered_curves = {}
    for algorithm in algorithms:
        complexity = df["Complexity"].copy().where(df["Type"] == algorithm).dropna().to_list()
        loyalty = df[metric].copy().where(df["Type"] == algorithm).dropna().to_list()

        if algorithm != "OS":
            # As OS vs rest of alg have opposite trends for the value of alpha, we need to reverse the order of the lists    
            complexity = complexity[::-1]
            loyalty = loyalty[::-1]

        filtered_complexity, filtered_loyalty = filter_anomalous_loyalty_curve(complexity, loyalty)
        
        if show_fig:
            plt.plot(complexity, loyalty)
            plt.plot(filtered_complexity, filtered_loyalty)
            plt.show()

        auc[algorithm] = np.trapz(filtered_loyalty, filtered_complexity)
        filtered_curves[algorithm] = (filtered_complexity, filtered_loyalty)

    return auc, filtered_curves
    

def filter_anomalous_loyalty_curve(x_values: List, y_values: List) -> tuple[List, List]:
    """
    Filter out anomalous behavior in loyalty vs complexity curves where there's an initial high loyalty followed by a decrease and then the expected pattern of increasing loyalty with complexity.
    The function uses slope analysis to find the point where the curve begins to consistently increase, which is considered the start of the valid data.
    """
    x = np.array(x_values)
    y = np.array(y_values)
    slopes = np.diff(y) / np.maximum(np.diff(x), 1e-10)   #Slope of curve at each point
    
    valid_idx = 0
    step = 7        
    for i in range(len(slopes) - step + 1):
        # Check if the majority of the next few slopes are positive
        if np.sum(slopes[i:i+step] > 0) >= 0.6*step:        # If we consider more % of slopes positive, then se delete sudden drops
            valid_idx = i
            break
    
    return x[valid_idx:].tolist(), y[valid_idx:].tolist()


def find_knee_curve(x_values: List, y_values: List) -> tuple[float, float]:
    """
    Find the knee point of the curve using the Kneedle algorithm from "Finding a “Kneedle” in a Haystack:Detecting Knee Points in System Behavior"
    https://github.com/arvkevi/kneed?tab=readme-ov-file#input-data
    """
    x = np.array(x_values)
    y = np.array(y_values)
    kneedle = KneeLocator(x, y, S=1.0, curve='concave', direction='increasing', online=True)
    knee_x = kneedle.knee
    knee_y = kneedle.knee_y
    return knee_x, knee_y


if __name__ == '__main__':
    #dataset = "Chinatown"
    dataset = "ItalyPowerDemand"
    models = ["cnn", "decision-tree", "logistic-regression", "knn"]
    for model in models:
        df = pd.read_csv(f"results/{dataset}/{model}_alpha_complexity_loyalty.csv")
        auc_value, filtered_tuple = auc(df, show_fig=False)
        knee_point = find_knee_curve(filtered_tuple[0], filtered_tuple[1])
        # Plot the filtered curve with the knee point
        plt.plot(filtered_tuple[0], filtered_tuple[1])
        plt.scatter(knee_point[0], knee_point[1], color='red')
        plt.show()