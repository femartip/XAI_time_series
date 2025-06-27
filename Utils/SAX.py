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
    first = True
    for i, ts_y in enumerate(time_series):
        n_timestamps = time_series.shape[1]
        ts_y = ts_y.reshape(1, -1)
        sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='normal')
        sax_y = sax.fit_transform(ts_y)[0]
        #bins = norm.ppf(np.linspace(0, 1, n_bins + 1)[1:-1])
        #bottom_bool = np.r_[True, sax_y[1:] > sax_y[:-1]]
        
        plt.figure(figsize=(6, 4))
        plt.plot(sax_y)
        #plt.plot(ts_y[0], 'o--', label='Original')
        #for x, y, s, bottom in zip(range(n_timestamps), ts_y[0], sax_y, bottom_bool):
            #va = 'bottom' if bottom else 'top'
            #plt.text(x, y, s, ha='center', va=va, fontsize=14, color='#ff7f0e')
        #plt.hlines(bins, 0, n_timestamps, color='g', linestyles='--', linewidth=0.5)
        #sax_legend = mlines.Line2D([], [], color='#ff7f0e', marker='*', label='SAX - {0} bins'.format(n_bins))
        #first_legend = plt.legend(handles=[sax_legend], fontsize=8, loc=(0.76, 0.86))
        #ax = plt.gca().add_artist(first_legend)
        #plt.legend(loc=(0.81, 0.93), fontsize=8)

        buf = io.BytesIO()
        if first:
            plt.savefig(f"./sax_{i}_{n_bins}.png")
            first = False
            
        plt.savefig(buf)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        ts_simplifications.append(f"data:image/png;base64,{img_b64}")
    
    return ts_simplifications

def main():
    Y = np.array([[-0.695489594455168, -0.8599324130362476, -0.985222179574213, -1.0478670628431956, -1.0772318518755313, -1.1359614299402025, -1.1711991767790053, -1.0713588940690641, -0.9480267801332545, -0.8051181401758877, -0.4860207660245071, 0.06995257298771432, 1.3620032904104824, 1.3287231961738353, 1.3267655435716796, 1.0996778417216173, 0.7610039415486797, 0.927404412731915, 1.1035931469259288, 0.9332773705383821, 0.4458218726016105, -0.06121015135671821, -0.5819457435301368, -0.8481864974233133]])
    
    for i in range(2,20):
        get_SAX(Y, n_bins=i)

if __name__ == "__main__":
    main()