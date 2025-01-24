import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score
from typing import List

from ORSalgorithm.Utils.loadModel import model_batch_classify
from ORSalgorithm.Perturbations.dataTypes import SegmentedTS
from ORSalgorithm.Utils.scoring_functions import score_simplicity

def calculate_mean_loyalty(batch_org_ts:List[float], batch_simplified_ts:List[SegmentedTS], model_path)->float:
    """
    Calculate Mean score to measure agreement between original and simplified classifications.
    """
    batch_simplified_ts = [ts.line_version for ts in batch_simplified_ts]
    pred_class_original = model_batch_classify(model_path, batch_of_timeseries=batch_org_ts) 
    pred_class_simplified = model_batch_classify(model_path, batch_of_timeseries=batch_simplified_ts)
    loyalty = np.mean(np.equal(pred_class_original, pred_class_simplified))
    return loyalty

def calculate_kappa_loyalty(batch_org_ts:List[float], batch_simplified_ts:List[SegmentedTS], model_path)->float:
    """
    Calculate Cohen's Kappa score to measure agreement between original and simplified classifications.
    """
    batch_simplified_ts = [ts.line_version for ts in batch_simplified_ts]
    pred_class_original = model_batch_classify(model_path, batch_of_timeseries=batch_org_ts) 
    pred_class_simplified = model_batch_classify(model_path, batch_of_timeseries=batch_simplified_ts)
    kappa_loyalty = cohen_kappa_score(pred_class_original, pred_class_simplified)
    return kappa_loyalty

def calculate_complexity(batch_simplified_ts: List[SegmentedTS])->float:
    """
    Calculate complexity of simplified time series as mean number of segments.
    """
    complexity = np.mean([score_simplicity(ts) for ts in batch_simplified_ts])
    return complexity