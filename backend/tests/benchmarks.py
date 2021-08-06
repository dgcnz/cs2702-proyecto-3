import unittest
from typing import List
from lib.rtree import RTree
from random import randrange
from random import randint
import numpy as np

Point = np.ndarray


def random_point(d: int) -> Point:
    MIN, MAX = -100, 100
    return np.array([randrange(MIN, MAX) for _ in range(d)])


def random_points(n: int, d: int) -> List[Point]:
    return [random_point(d) for _ in range(n)]


def rtree_knn(rtree):
    c = random_point(128)
    rtree.knn(c, 8, False)


def test_benchmark10(benchmark):
    N = 10
    points = random_points(N, 128)
    rtree = RTree(128, False)
    rtree.build(points)
    benchmark(rtree_knn, rtree)


def test_benchmark100(benchmark):
    N = 100
    points = random_points(N, 128)
    rtree = RTree(128, False)
    rtree.build(points)
    benchmark(rtree_knn, rtree)
