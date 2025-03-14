from typing import List, TypedDict, Dict, Any
from heapq import heapify, heappop

from SimplificationMethods.BottumUp.Heap import new_heapify,new_heappop, peek
from Utils.dataTypes import SegmentedTS
import numpy as np

class Segment:
    def __init__(self, x_1:int, x_2:int, y_1:float, y_2:float):
        self.start_x = x_1
        self.end_x = x_2
        self.start_y = y_1
        self.end_y = y_2
        if x_1 == x_2:
            self.m = float('inf')
        else:
            self.m = (y_2 - y_1) / (x_2 - x_1)
        self.b =  y_1-self.m*x_1

    def error(self, x_tests, y_tests):
        self_y = [self.m*x+self.b for x in x_tests]
        return np.sum([abs(y_self-y_test) for y_self, y_test in zip(self_y, y_tests)])

    def evaluate_at(self,x):
        if x==self.start_x:
            return self.start_y
        elif x==self.end_x:
            return self.end_y

        return x*self.m+self.b

    def __repr__(self):
        return f"({self.start_x}, {self.start_y})--( {self.end_x}, {self.end_y})"

def merge(segment1 :Segment, segment2:Segment)->Segment:
    """
    We merge to segments by making a straight line from the start of the left most segment to the end of the right most segment
    """
    if segment1.start_x < segment2.start_x:
        leftmost_x = segment1.start_x
        leftmost_y = segment1.evaluate_at(leftmost_x)
        rightmost_x = segment2.end_x
        rightmost_y = segment2.evaluate_at(rightmost_x)
    else:
        leftmost_x = segment2.start_x
        leftmost_y = segment2.evaluate_at(leftmost_x)
        rightmost_x = segment1.end_x
        rightmost_y = segment1.evaluate_at(rightmost_x)

    newSegment = Segment(x_1=leftmost_x, x_2=rightmost_x, y_1=leftmost_y, y_2=rightmost_y)
    return newSegment

def merge_and_score_error(segment1:Segment,segment2:Segment, ts):
    mergeSegs = merge(segment1, segment2)
    x_test = list(range(mergeSegs.start_x, mergeSegs.end_x + 1))
    y_test = [ts[i] for i in x_test]
    merge_cost = mergeSegs.error(x_tests=x_test, y_tests=y_test)
    return merge_cost, mergeSegs


def bottom_up(ts:List[float], max_error):
    segTS = []
    for i in range(len(ts)):
        x1=i
        y1=ts[x1]
        segTS.append(Segment(x_1=x1, x_2=x1, y_1=y1, y_2=y1))

    mergeCosts : List[float] = []

    for i in range(len(segTS)-1):
        error, mergeSeg = merge_and_score_error(segTS[i], segTS[i+1], ts=ts)
        mergeCosts.append(error)




    while len(mergeCosts) >1 and min(mergeCosts) < max_error:
        i = mergeCosts.index(min(mergeCosts))
        segTS[i] = merge(segTS[i], segTS[i+1]) # Overwrite with new bigger segment

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
    ts = [0,20,0,1,5,20,20,20,5,4,3,2,1]
    segments = bottom_up(ts, 10)
    for seg in segments:
        print(seg,end="\t")

    print()
    print(convert_to_segmentedTS(segments=segments, ts_length=len(ts)))




