from heapq import heapify, heappop
from functools import cmp_to_key
from typing import List, Any

def new_heapify(data, cmp):
    s = list(map(cmp_to_key(cmp), data))
    heapify(s)
    return s

def new_heappop(data):
    return heappop(data).obj

def peek(heap: List[Any]) -> Any:
    if heap:
        return heap[0].obj
    raise IndexError("peek from an empty heap")