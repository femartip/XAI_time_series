from ORSalgorithm.Utils.types import HeapStruct
from typing import Any


class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, val: HeapStruct):
        self.heap.append(val)
        self.__percolateUp(len(self.heap)-1)

    def getMin(self) -> HeapStruct | None:
        if self.heap:
            return self.heap[0]
        return None

    def removeMin(self) -> HeapStruct | None:
        if len(self.heap) > 1:
            min = self.heap[0]
            self.heap[0] = self.heap[-1]
            del self.heap[-1]
            self.__minHeapify(0)
            return min
        elif len(self.heap) == 1:
            min = self.heap[0]
            del self.heap[0]
            return min
        else:
            return None

    def __percolateUp(self, index):
        parent = (index-1)//2
        if index <= 0:
            return
        elif self.heap[parent].error > self.heap[index].error:
            tmp = self.heap[parent]
            self.heap[parent] = self.heap[index]
            self.heap[index] = tmp
            self.__percolateUp(parent)

    def __minHeapify(self, index):
        left = (index * 2) + 1
        right = (index * 2) + 2
        smallest = index
        if len(self.heap) > left and self.heap[smallest].error > self.heap[left].error:
            smallest = left
        if len(self.heap) > right and self.heap[smallest].error > self.heap[right].error:
            smallest = right
        if smallest != index:
            tmp = self.heap[smallest]
            self.heap[smallest] = self.heap[index]
            self.heap[index] = tmp
            self.__minHeapify(smallest)

if __name__ == "__main__":
    heap = MinHeap()
    heap.insert(HeapStruct(12, 'a', 1, 'b', 2, 'seg1'))
    heap.insert(HeapStruct(10, 'c', 3, 'd', 4, 'seg2'))
    heap.insert(HeapStruct(-10, 'e', 5, 'f', 6, 'seg3'))
    heap.insert(HeapStruct(100, 'g', 7, 'h', 8, 'seg4'))

    print(heap.getMin())
    print(heap.removeMin())
    print(heap.removeMin())

    print(heap.removeMin())
    print(heap.removeMin())
