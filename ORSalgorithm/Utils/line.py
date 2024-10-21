from typing import List, Tuple
import numpy as np


def euclidean_distance_weighted(ts1: List[float], ts2: list[float], weight: float) -> float:
    # Euclidean distance

    ts1 = np.array(ts1)
    ts2 = np.array(ts2)
    dist = np.linalg.norm(ts1 - ts2)

    return dist / weight


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


def convert_all_points_to_lines(ts_length: int, all_x_selected: List[List[int]], all_y_selected: List[List[float]]) -> \
        List[List[float]]:
    all_line_versions_y = []
    for x_selected, y_selected in zip(all_x_selected, all_y_selected):
        line_version = interpolate_points_to_line(ts_length=ts_length, x_selected=x_selected, y_selected=y_selected)
        all_line_versions_y.append(line_version)
    return all_line_versions_y


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


def check_if_all_is_on_line(x1, y1, x2, y2, x3, y3):
    if (y1 == y2 and x1 == x2) or (y1 == y3 and x1 == x3) or (y2 == y3 and x2 == x3):
        return True
    tolerance = 1e-9  # Floating Point Error, adjust this value as needed
    return abs(calculate_line_equation(x1, y1, x2, y2, x3) - y3) < tolerance


def get_pivot_points(approximation: List[float], x_and_y=False) -> List[int] | Tuple[List[float], List[float]]:
    """
    If x_and_y is False, returns the pivot points of the approximation.
    :param approximation:
    :param x_and_y:
    :return:
    """
    just_pivot_points_x = [0]  # We will always have the first point
    just_pivot_points_y = [approximation[0]]
    prev_x = 0
    prev_y = approximation[0]
    possible_next_x = 1
    possible_next_y = approximation[1]
    for i in range(2, len(approximation)):
        curr_x = i
        curr_y = approximation[i]
        if check_if_all_is_on_line(prev_x, prev_y, possible_next_x, possible_next_y, curr_x, curr_y):
            possible_next_x = curr_x
            possible_next_y = curr_y
        else:
            just_pivot_points_x.append(possible_next_x)
            just_pivot_points_y.append(possible_next_y)
            prev_x = possible_next_x
            prev_y = possible_next_y
            possible_next_x = curr_x
            possible_next_y = curr_y

    just_pivot_points_x.append(possible_next_x)
    just_pivot_points_y.append(possible_next_y)

    if x_and_y:
        return just_pivot_points_x, just_pivot_points_y
    return just_pivot_points_x


if __name__ == '__main__':
    ys = [0, 0, 1, 2, 3, 5, 7, 9, 11, 5, -1, 5, 5, 7]
    print(get_pivot_points(ys))
    ys = [0, 0, 0, 0, 0, 0, 0, 0]
    print(get_pivot_points(ys))
    ys = [0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
    print(get_pivot_points(ys))
