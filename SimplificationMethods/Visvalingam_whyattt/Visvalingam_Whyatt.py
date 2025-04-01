
from simplification.cutil import (
    simplify_coords,
    simplify_coords_idx,
    simplify_coords_vw,
    simplify_coords_vw_idx,
    simplify_coords_vwp,
)

# From this place https://github.com/urschrei/simplification/tree/master
"""
@software{Hugel_Simplification_2021,
author = {Hügel, Stephan},
doi = {10.5281/zenodo.5774852},
license = {MIT},
month = {12},
title = {{Simplification}},
url = {https://github.com/urschrei/simplification},
version = {X.Y.Z},
year = {2021}
}
"""
import numpy as np
from SimplificationMethods.ORSalgorithm.Perturbations.dataTypes import SegmentedTS

def simplify(ts, alpha):
    alpha = alpha
    x_values = list(range(len(ts)))
    coords_vw = np.array(list(zip(x_values, ts)))
    simp = simplify_coords_vw(coords_vw, alpha)
    x_values = [x_val for x_val,y_val in simp]
    y_values = [y_val for x_val,y_val in simp]
    segTS = SegmentedTS(x_pivots=x_values,y_pivots=y_values,ts_length=len(ts))
    return segTS

if __name__ == "__main__":
    pass
