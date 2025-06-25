from Utils.dataTypes import SegmentedTS
from pyts.approximation import SymbolicAggregateApproximation
from scipy.stats import norm
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import base64
import io

def get_SAX(time_series: np.ndarray, n_bins:int = 3) -> list[str]:
    """
    Apply SAX algorithm for all time series in the dataset.
    """
    ts_simplifications = []
    
    for i, ts_y in enumerate(time_series):
        n_timestamps = time_series.shape[1]
        ts_y = ts_y.reshape(1, -1)
        sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='normal')
        sax_y = sax.fit_transform(ts_y)[0]
        bins = norm.ppf(np.linspace(0, 1, n_bins + 1)[1:-1])
        bottom_bool = np.r_[True, sax_y[1:] > sax_y[:-1]]

        
        plt.figure(figsize=(6, 4))
        plt.plot(ts_y[0], 'o--', label='Original')
        for x, y, s, bottom in zip(range(n_timestamps), ts_y[0], sax_y, bottom_bool):
            va = 'bottom' if bottom else 'top'
            plt.text(x, y, s, ha='center', va=va, fontsize=14, color='#ff7f0e')
        plt.hlines(bins, 0, n_timestamps, color='g', linestyles='--', linewidth=0.5)
        sax_legend = mlines.Line2D([], [], color='#ff7f0e', marker='*', label='SAX - {0} bins'.format(n_bins))
        first_legend = plt.legend(handles=[sax_legend], fontsize=8, loc=(0.76, 0.86))
        ax = plt.gca().add_artist(first_legend)
        plt.legend(loc=(0.81, 0.93), fontsize=8)
        plt.xlabel('Time', fontsize=14)

        buf = io.BytesIO()
        
        #plt.savefig(f"./sax_{i}.png")
            
        plt.savefig(buf)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        ts_simplifications.append(f"data:image/png;base64,{img_b64}")
    
    return ts_simplifications

def main():
    X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Y = np.array([[6, 3, 3, 5, 8, 6, 6, 7, 8, 9.1, 10]])
    
    #simp = get_SAX(Y)

if __name__ == "__main__":
    main()