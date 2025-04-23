import statsmodels
from statsmodels.tsa.stattools import adfuller, acf
import antropy as ant
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def adf_test(ts: np.ndarray) -> float:
    result = adfuller(ts)
    p_value = float(result[1])
    return p_value

def acf_test(ts: np.ndarray, show: bool = True) -> float:
    result = acf(ts, fft=True)
    autocorr = float(np.mean(np.abs(result[2:])))
    if show and autocorr > 0.35 and autocorr < 0.45:
        label = True if autocorr > 0.4 else False
        print(f"Autocorr: {label}")
        plt.plot(ts)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    return autocorr

#https://en.wikipedia.org/wiki/Approximate_entropy
def get_entropy(ts: np.ndarray) -> float:
    entropy = ant.app_entropy(ts)
    return entropy

if __name__ == '__main__':
    metadata = pd.read_csv("./data/DataSummary.csv")
    datasets = metadata["Name"].unique().tolist()
    #datasets = ["Chinatown"]

    is_stationary = {}
    is_seasonal = {}
    is_entropy = {}

    for dataset in datasets:
        print(f"Processing dataset {dataset}".upper())
        
        data = np.load(f"./data/{dataset}/{dataset}_TRAIN.npy")
        labels = data[:,0]
        data = data[:,1:]
        print(f"\tShape of dataset: {data.shape}")

        #Stationarity https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
        p_values = [adf_test(x) for x in data if not np.any(np.isnan(x))]
        n_stationary = sum([p < 0.05 for p in p_values])
        stationary_pct = n_stationary / len(p_values) * 100
        if stationary_pct > 80:
            stationary_label = "True"
        elif stationary_pct < 80 and stationary_pct > 50:
            stationary_label = "Partial"
        else:
            stationary_label = "False"
        
        is_stationary[dataset] = stationary_label
        print(f"\t{stationary_pct:.1f}% of time series are stationary (ADF test p < 0.05)")
        print(f"\tWe consider it {stationary_label} stationary")
        
        autocorr_coeff = [acf_test(x, show=False) for x in data if not np.any(np.isnan(x))]
        seasonal = [True if x>0.4 else False for x in autocorr_coeff]
        seasonal_labels = Counter(seasonal)
        seasonal_label_dict = seasonal_labels.most_common(1)[0]
        seasonal_label = seasonal_label_dict[0]
        seasonal_pct = seasonal_label_dict[1]/len(seasonal) * 100

        is_seasonal[dataset] = seasonal_label
        print(f"\t{seasonal_pct:.1f}% of time series are {'seasonal' if seasonal_label else 'not seasonal'}")
        print(f"\tWe consider it {seasonal_label} seasonal")

        entropies = [get_entropy(x) for x in data if not np.any(np.isnan(x))]
        entropy = np.mean(entropies)
        entropy_std = np.std(entropies)
        is_entropy[dataset] = entropy
        print(f"\tEntropy of dataset {entropy}+-{entropy_std}")

    metadata["Stationary"] = metadata["Name"].map(is_stationary)
    metadata["Seasonal"] = metadata["Name"].map(is_seasonal)
    metadata["Entropy"] = metadata["Name"].map(is_entropy)
    metadata.to_csv("./data/DataSummary.csv", index=False)
    print("Metadata saved")
    print(metadata.head())



