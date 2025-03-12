import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score
from typing import List

from Utils.dataTypes import SegmentedTS

def score_simplicity(approximation: SegmentedTS) -> float:
        simplicity = (len(approximation.x_pivots) - 1)  * (1 / (len(approximation.line_version) - 1))
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