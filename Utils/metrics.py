import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score
import pandas as pd
from kneed import KneeLocator
import pandas as pd

from Utils.dataTypes import SegmentedTS 

def score_simplicity(approximation: SegmentedTS) -> float:
    if approximation.num_real_segments is None:
        simplicity = (len(approximation.x_pivots) - 1)  * (1 / (len(approximation.line_version) - 1))
    else:
        simplicity = approximation.num_real_segments  * (1 / (len(approximation.line_version) - 1))
        
    return simplicity

def calculate_mean_loyalty(pred_class_original:list[int], pred_class_simplified:list[int])->float:
    """
    Calculate Mean score to measure agreement between original and simplified classifications.
    """
    loyalty = np.mean(np.equal(pred_class_original, pred_class_simplified))
    loyalty = float(loyalty)
    return loyalty

def calculate_kappa_loyalty(pred_class_original:list[int], pred_class_simplified:list[int], num_classes)->float:
    """
    Calculate Cohen's Kappa score to measure agreement between original and simplified classifications.
    """
    #https://github.com/scikit-learn/scikit-learn/issues/9624 
    if len(set(pred_class_original).union(set(pred_class_simplified))) == 1:
        kappa_loyalty = 1.0
    else:
        list_classes = [i for i in range(num_classes)]
        kappa_loyalty = cohen_kappa_score(pred_class_original, pred_class_simplified, labels=list_classes)
    
    return kappa_loyalty

def calculate_percentage_agreement(pred_class_original:list[int], pred_class_simplified:list[int]) -> int:
    """
    Calculate percentage of agreement between original and simplified classifications.
    """
    pa = sum(np.array(pred_class_original) == np.array(pred_class_simplified)) / len(pred_class_simplified)
    pa = int(pa * 100)
    return pa

def calculate_complexity(batch_simplified_ts: list[SegmentedTS])->float:
    """
    Calculate complexity of simplified time series as mean number of segments.
    """
    scores = [score_simplicity(ts) for ts in batch_simplified_ts]
    complexity = sum(scores) / len(scores) if len(scores) > 0 else 0.0
    return complexity

def auc(df: pd.DataFrame, metric:str="Kappa Loyalty", show_fig:bool=False) -> tuple[dict[str, float], dict[str,tuple[list, list]]]:
    """
    Calculate the Area Under the Curve of the Complexity vs Loyalty curve for each simplification algorithm.
    This is used to compare the performance of the different simplification algorithms.
    """
    assert metric != "Kappa Loyalty" or metric != "Mean Loyalty", "Metric must be either Kappa Loyalty or Mean Loyalty"

    algorithms = df["Type"].unique()
    all_auc = {}
    filtered_curves = {}
    for algorithm in algorithms:
        complexity = df["Complexity"].copy().where(df["Type"] == algorithm).dropna().to_list()
        loyalty = df[metric].copy().where(df["Type"] == algorithm).dropna().to_list()

        sort_id = sorted(range(len(complexity)), key=lambda x: complexity[x])
        complexity = [complexity[x] for x in sort_id]
        loyalty = [loyalty[x] for x in sort_id]

        if loyalty[-1] != 1 or complexity[-1] != 1:
            complexity.append(1)
            loyalty.append(1)

        filtered_complexity, filtered_loyalty = filter_anomalous_loyalty_curve(complexity, loyalty)

        auc = 0.0
        for i in range(len(complexity) -1):
            x_1 = complexity[i]
            x_2 = complexity[i+1]

            y_1 = loyalty[i]
            y_2 = loyalty[i+1]

            auc += abs(((x_2 - x_1)*(y_1 + y_2))/2)

        all_auc[algorithm] = auc
        filtered_curves[algorithm] = (filtered_complexity, filtered_loyalty)

        if show_fig:
            print("AUC",auc)
            plt.plot(complexity, loyalty)
            #plt.plot(filtered_complexity, filtered_loyalty)
            plt.title(f"{algorithm} - {metric}")
            plt.xlabel("Complexity")
            plt.ylabel("Loyalty")
            plt.legend(["Original", "Filtered"])
            plt.show()

    return all_auc, filtered_curves
    

def filter_anomalous_loyalty_curve(x_values: list, y_values: list) -> tuple[list, list]:
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


def find_knee_curve(x_values: list, y_values: list) -> tuple[float, float]:
    """
    Find the knee point of the curve using the Kneedle algorithm from "Finding a “Kneedle” in a Haystack:Detecting Knee Points in System Behavior"
    https://github.com/arvkevi/kneed?tab=readme-ov-file#input-data
    """
    x = np.array(x_values)
    y = np.array(y_values)
    kneedle = KneeLocator(x, y, S=1.0, curve='concave', direction='increasing', online=True)
    knee_x = kneedle.knee
    knee_y = kneedle.knee_y
    if knee_y is None or knee_x is None:
        knee_x = x[-1]
        knee_y = y[-1]
    return knee_x, knee_y


def update_auc(results_df: pd.DataFrame) -> None:
    datasets_names = results_df["dataset"].unique().tolist()
    datasets = [dataset.replace("TEST_normalized","") for dataset in datasets_names]

    #datasets = ['ProximalPhalanxOutlineCorrect', 'ItalyPowerDemand', 'MoteStrain', 'GunPointOldVersusYoung', 'MiddlePhalanxTW', 'ECG200', 'SonyAIBORobotSurface1', 'ElectricDevices', 'BME', 'Chinatown', 'DistalPhalanxOutlineAgeGroup', 'MedicalImages', 'TwoPatterns', 'UMD', 'ECG5000', 'TwoLeadECG', 'GunPointAgeSpan', 'MiddlePhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxTW', 'SmoothSubspace', 'Plane', 'MiddlePhalanxOutlineCorrect', 'Adiac', 'SwedishLeaf', 'ECGFiveDays', 'PhalangesOutlinesCorrect', 'FacesUCR', 'CBF', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'Wafer']
    #models = ["cnn", "decision-tree", "logistic-regression", "knn"]
    models = ["cnn"]
    rows_to_drop = []
    rows_to_update=[]
    for i, dataset in enumerate(datasets):
        print(f"Dataset {dataset}")
        for model in models:
            print(f"Model {model}")
            df = pd.read_csv(f"results/{dataset}/{model}_alpha_complexity_loyalty.csv")
            auc_values, filtered_curves = auc(df, show_fig=False)
            comp_threshold = get_loylaty_by_threshold(df, 0.8)
                

            simp_alg = ["OS", "RDP", "VW", "BU_1", "BU_2"]

            for alg in simp_alg:
                query_mask = ((results_df["dataset"] == f"{dataset}TEST_normalized") & (results_df["model"] == f"{model}_norm.pth") & (results_df["simp_algorithm"] == alg))
                
                if query_mask.any(): rows_to_drop.extend(results_df[query_mask].index.tolist())
                
                new_row = {"dataset": f"{dataset}TEST_normalized","model": f"{model}_norm.pth","simp_algorithm": alg,"performance": auc_values[alg],"comp@loy=0.8": comp_threshold[alg],"time": 0.0}
                rows_to_update.append(new_row)
    
    if rows_to_drop: results_df = results_df.drop(index=rows_to_drop)
    
    new_rows_df = pd.DataFrame(rows_to_update)
    results_df = pd.concat([results_df, new_rows_df], ignore_index=True)

        
    results_df.to_csv("./results/results_copy.csv", index=False)
    print("done")

def get_loylaty_by_threshold(df: pd.DataFrame, loyalty_threshold: float, metric:str="Kappa Loyalty") -> dict[str, float]:
    algorithms = df["Type"].unique()
    threshold_comp = {}
    
    for algorithm in algorithms:
        complexity = df["Complexity"].copy().where(df["Type"] == algorithm).dropna().to_list()
        loyalty = df[metric].copy().where(df["Type"] == algorithm).dropna().to_list()

        sort_id = sorted(range(len(complexity)), key=lambda x: complexity[x])
        complexity = [complexity[x] for x in sort_id]
        loyalty = [loyalty[x] for x in sort_id]
        
        if loyalty[-1] != 1 or complexity[-1] != 1:
            complexity.append(1)
            loyalty.append(1)

        #print(min(complexity), max(complexity))

        if loyalty_threshold in loyalty:
            threshold_idx = loyalty.index(loyalty_threshold)
            threshold_comp[algorithm] = complexity[threshold_idx]
        else:
            interpolated_comp  = 1.0
            for i in range(len(complexity)-1):
                if loyalty[i] < loyalty_threshold and loyalty[i+1] > loyalty_threshold:
                    interpolated_comp = np.interp(x=0.8,xp=[loyalty[i], loyalty[i+1]], fp=[complexity[i], complexity[i+1]])
                    threshold_comp[algorithm] = interpolated_comp
                    if interpolated_comp > 1.0: 
                        print(f"Interpolated value {interpolated_comp} > 1.0")
                        interpolated_comp = 1.0
                    break
                elif loyalty[i] > loyalty_threshold and loyalty[i+1] > loyalty_threshold:
                    interpolated_comp = 1.0
                    threshold_comp[algorithm] = interpolated_comp

    return threshold_comp


if __name__ == '__main__':
    from plotting import plot_csv_complexity_kappa_loyalty
    datasets = ['ProximalPhalanxOutlineCorrect', 'ItalyPowerDemand', 'MoteStrain', 'GunPointOldVersusYoung', 'MiddlePhalanxTW', 'ECG200', 'SonyAIBORobotSurface1', 'ElectricDevices', 'BME', 'Chinatown', 'DistalPhalanxOutlineAgeGroup', 'MedicalImages', 'TwoPatterns', 'UMD', 'ECG5000', 'TwoLeadECG', 'GunPointAgeSpan', 'MiddlePhalanxOutlineAgeGroup', 'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxTW', 'SmoothSubspace', 'Plane', 'MiddlePhalanxOutlineCorrect', 'Adiac', 'SwedishLeaf', 'ECGFiveDays', 'PhalangesOutlinesCorrect', 'FacesUCR', 'CBF', 'DistalPhalanxOutlineCorrect', 'DistalPhalanxTW', 'Wafer']
    #datasets = ['MoteStrain', 'ECG5000', 'ECGFiveDays', 'Wafer']
    model = "cnn"
    for dataset in datasets:
        print(dataset)
        results_file = f"results/{dataset}/{model}_alpha_complexity_loyalty.csv"
        df = pd.read_csv(results_file)
        values = get_loylaty_by_threshold(df, 0.8)
        #for alg in values:
        #    if values[alg] > 1:
        #        print(dataset)
        #        print(alg)
        #        print(values[alg])
        #fig = plot_csv_complexity_kappa_loyalty(results_file)
        #plt.show()
        #plt.show(block=False)
        #plt.pause(3)
        #plt.close()

    
    if False:
        results_df = pd.read_csv("./results/results_copy.csv")
        update_auc(results_df=results_df)

