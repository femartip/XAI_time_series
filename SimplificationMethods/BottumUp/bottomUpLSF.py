from typing import List, TypedDict, Dict, Any
from heapq import heapify, heappop

from SimplificationMethods.BottumUp.Heap import new_heapify,new_heappop, peek
from Utils.dataTypes import SegmentedTS
import numpy as np


def best_fit_line(ts, xi, xj):
    """
    Find the best fit line for a subset of time series ts from index xi to xj (inclusive).
    Returns slope (m) and y-intercept (b).
    """
    # Extract the subset of y values
    y_subset = ts[xi:xj + 1]

    # Create corresponding x values (indices)
    x_subset = list(range(xi, xj + 1))

    n = len(x_subset)

    # Calculate sums needed for the least squares formula
    sum_x = sum(x_subset)
    sum_y = sum(y_subset)
    sum_xy = sum(x * y for x, y in zip(x_subset, y_subset))
    sum_x_squared = sum(x ** 2 for x in x_subset)

    # Calculate slope using the least squares formula
    # m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x**2)
    numerator = n * sum_xy - sum_x * sum_y
    denominator = n * sum_x_squared - sum_x ** 2

    # Handle potential division by zero
    if denominator == 0:
        m = 0  # Horizontal line for perfectly vertical data
    else:
        m = numerator / denominator

    # Calculate y-intercept
    # b = (sum_y - m * sum_x) / n
    b = (sum_y - m * sum_x) / n

    return m, b


class Segment:
    def __init__(self, x_1:int, x_2:int,m:float,b:float):
        self.start_x = x_1
        self.end_x = x_2
        self.m = m
        self.b = b

    def error(self, x_tests, y_tests):
        self_y = [self.m*x+self.b for x in x_tests]
        return np.sum([abs(y_self-y_test) for y_self, y_test in zip(self_y, y_tests)])

    def evaluate_at(self,x):
        return x*self.m+self.b

    def __repr__(self):
        return f"({self.start_x}, {self.evaluate_at(self.start_x)})--( {self.end_x}, {self.evaluate_at(self.end_x)})"

def merge(segment1 :Segment, segment2:Segment, ts)->Segment:
    """
    We merge two segments by making a Least Square Fit over all y in the area
    """
    if segment1.start_x < segment2.start_x:
        leftmost_x = segment1.start_x
        rightmost_x = segment2.end_x
        m,b = best_fit_line(ts=ts,xi=leftmost_x,xj=rightmost_x)
    else:
        leftmost_x = segment2.start_x
        rightmost_x = segment1.end_x
        m, b = best_fit_line(ts=ts, xi=leftmost_x, xj=rightmost_x)

    newSegment = Segment(x_1=leftmost_x, x_2=rightmost_x,m=m,b=b)
    return newSegment

def merge_and_score_error(segment1:Segment,segment2:Segment, ts):
    mergeSegs = merge(segment1, segment2,ts)
    x_test = list(range(mergeSegs.start_x, mergeSegs.end_x + 1))
    y_test = [ts[i] for i in x_test]
    merge_cost = mergeSegs.error(x_tests=x_test, y_tests=y_test)
    return merge_cost, mergeSegs


def bottom_up(ts:List[float], max_error):
    segTS = []
    for i in range(len(ts)):
        x1=i
        y1=ts[x1]
        segTS.append(Segment(x_1=x1, x_2=x1, m=0,b=x1))

    mergeCosts : List[float] = []

    for i in range(len(segTS)-1):
        error, mergeSeg = merge_and_score_error(segTS[i], segTS[i+1], ts=ts)
        mergeCosts.append(error)




    while len(mergeCosts) >1 and min(mergeCosts) < max_error:
        i = mergeCosts.index(min(mergeCosts))
        segTS[i] = merge(segTS[i], segTS[i+1],ts) # Overwrite with new bigger segment

        # Delete the joint segment
        segTS = segTS[:i+1] + segTS[i+2:]
        mergeCosts = mergeCosts[:i+1] + mergeCosts[i+2:]

        # Combine ahead
        if i +1  < len(segTS):
            error_ahead, mergeSeg =  merge_and_score_error(segTS[i], segTS[i+1], ts=ts)
            mergeCosts[i] = error_ahead
        else:
            mergeCosts = mergeCosts[:i]

        # Combine before
        if i >= 1:
            error_behind, mergeSeg = merge_and_score_error(segTS[i-1], segTS[i], ts=ts)
            mergeCosts[i-1] = error_behind


    return segTS



def convert_to_segmentedTS(segments:List[Segment], ts_length) -> SegmentedTS:
    pivot_x = []
    pivot_y = []
    num_segments = len(segments)
    for segment in segments:
        start_x = segment.start_x
        start_y = segment.evaluate_at(start_x)
        pivot_x.append(start_x)
        pivot_y.append(start_y)

        if segment.start_x != segment.end_x:
            end_x = segment.end_x
            end_y = segment.evaluate_at(end_x)
            pivot_x.append(end_x)
            pivot_y.append(end_y)

    return SegmentedTS(pivot_x, pivot_y, ts_length=ts_length, num_real_segments=num_segments)


def get_swab_approx(ts, max_error):
    max_error = max_error*10 # We enforce it to be a bit higher
    get_segments = bottom_up(ts=ts,max_error=max_error)
    segTS = convert_to_segmentedTS(get_segments, ts_length=len(ts))
    return segTS


if __name__ == "__main__":
    ts = [0,20,0,1,5,20,20,20,1.9,2+10**3,2+2*10**3,2.1+3*10**3]
    segments = bottom_up(ts, 10)
    for seg in segments:
        print(seg,end="\t")

    print()
    print(convert_to_segmentedTS(segments=segments, ts_length=len(ts)))




