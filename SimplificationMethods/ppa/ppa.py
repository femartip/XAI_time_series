from pyts.approximation import SymbolicAggregateApproximation
from Utils.load_data import load_dataset
from matplotlib import pyplot as plt
def get_simplification_paa(ts, num_windows,verbose=False):
    import numpy as np
    from pyts.approximation import PiecewiseAggregateApproximation

    # Parameters
    n_samples, n_timestamps = 1, len(ts)

    # PAA transformation
    n_windows = num_windows
    window_size = n_timestamps // n_windows
    paa = PiecewiseAggregateApproximation(window_size=window_size)
    ts_paa = paa.transform([ts])
    bounds = np.linspace(0, len(ts), n_windows + 1).astype('int64')
    start = bounds[:-1]
    end = bounds[1:]
    ts_paa = ts_paa[0]
    ppa_simp = [0]*len(ts)
    x_pivots = []
    y_pivots = []
    for t,s,e in zip(ts_paa,start,end):

        for i in range(s,e):
            ppa_simp[i] = t


        x_pivots.append(s)
        y_pivots.append(t)
        x_pivots.append(e-1)
        y_pivots.append(t)





    if verbose:
        plt.scatter( x_pivots,y_pivots, color="blue", label="approximation")
        plt.plot(ts, color="black", label="original")
        plt.plot(ppa_simp, color="red", label="simplified")
        plt.legend()
        plt.show()
    return ppa_simp, x_pivots


if __name__ == "__main__":
    dataset_name = "Chinatown"
    data = load_dataset(dataset_name=dataset_name, data_type="TRAIN_normalized")
    windows = 5
    paa_pivots = []
    for ts in data:
        paa_simp, x_pivots = get_simplification_paa(ts,num_windows=windows,verbose=True)
        paa_pivots.append(x_pivots)

    from Utils.load_data import load_dataset
    from matplotlib import pyplot as plt
    for ts, pivot in zip(data[:3], paa_pivots[:3]):
        plt.scatter(pivot,ts[pivot], label="Pivot points" )
        plt.plot(ts, label="Org")
        plt.legend()

        plt.show()
