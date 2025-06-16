from rdp import rdp
from typing import List
from simplifications import get_OS_simplification, get_RDP_simplification, get_bottom_up_simplification, \
    get_VW_simplification, get_LSF_simplification

import numpy as np
def calculate_line_equation(x1, y1, x2, y2, x3):
    # Calculate the slope (m)
    delta_x = x2 - x1
    if delta_x == 0:
        raise ValueError("The points must have different x-coordinates to calculate the slope.")
    m = (y2 - y1) / delta_x

    # Calculate the y-intercept (b)
    b = y1 - m * x1

    # Calculate y3 at x3
    y3 = m * x3 + b

    return y3

def interpolate_points_to_line(ts_length: int, x_selected: List[int], y_selected: List[float]) -> List[float]:
    """
    Given a list (points) of [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] of selected points calculate the y value of
    each timeStep.

    For each x in range(timeStep) we have 3 cases:
    1. x1 <= x <= x4: Find the pair xi <= x <=xi+1, s.t. i<=3. Use this slope to find the corresponding y value.
    2. x < x1. Extend the slope between x1 and x2 to x, and find the corresponding y value.
    3. x4 < x. Extend the slope between x3 and x4 to x, and find the corresponding y value.
    :param ts_length:
    :param y_selected:
    :param x_selected:
    :return:
    """

    interpolation_ts = [0 for _ in range(ts_length)]
    pointsX = 0
    for x in range(ts_length):
        # If x is bigger than x_selected[pointsX+1] we are in the next interval
        # pointsX < len(x_selected) - 2 Indicates that we extrapolate the two last points even if x is after this.
        if x > x_selected[pointsX + 1] and pointsX < len(x_selected) - 2:
            pointsX += 1

        x1 = x_selected[pointsX]
        x2 = x_selected[pointsX + 1]
        y1 = y_selected[pointsX]
        y2 = y_selected[pointsX + 1]
        x3 = x
        y3 = calculate_line_equation(x1, y1, x2, y2, x3)
        interpolation_ts[x] = y3

    return interpolation_ts


def find_alpha_giving_target_loyalty(loyalty,dataset_name, algo):
    import pandas as pd
    df = pd.read_csv(f"results/{dataset_name}/cnn_alpha_complexity_loyalty.csv")
    df_algo = df[df["Type"] == algo]
    df_threshold = df_algo[df_algo["Percentage Agreement"] >= loyalty]
    df_min_row_idx = df_threshold["Complexity"].idxmin()

    min_row = df.loc[[df_min_row_idx]]
    return min_row["Alpha"].tolist()[0]




def simplify_ts_by_loyalty(algo,loyalty, time_series, dataset_name):
    algo = algo.upper()
    alpha = find_alpha_giving_target_loyalty(loyalty,dataset_name,algo)
    print("alpha:", alpha)
    return simplify_ts_by_alpha(algo,alpha,time_series)


def simplify_ts_by_alpha(algo, alpha, time_series):
    algo = algo.upper()
    all_ts = [time_series]
    all_ts = np.array(all_ts)
    if algo == "OS":
        simp = get_OS_simplification(all_ts, alpha)
    elif algo == "RDP":
        simp = get_RDP_simplification(all_ts, alpha)
    elif algo == "BOTTOM-UP":
        simp = get_bottom_up_simplification(all_ts, alpha)
    elif algo == "VW":
        simp = get_VW_simplification(all_ts, alpha)
    elif algo == "LSF":
        simp = get_LSF_simplification(all_ts, alpha)
    else:
        raise ValueError("Unknown algorithm '{}'".format(algo))

    selected_simp = simp[0]
    return selected_simp.line_version



if __name__ == '__main__':
    import random
    random.seed(42)
    ts_org = [random.randint(1,100)/100 for _ in range(24)]
    ts = get_simplification(ts_org)

    import matplotlib.pyplot as plt
    xs = list(range(len(ts)))
    plt.plot(xs,ts, linestyle='--',  color='r')
    plt.plot(list(range(len(ts_org))),ts_org)
    plt.show()

