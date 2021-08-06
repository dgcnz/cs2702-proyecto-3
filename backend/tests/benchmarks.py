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


def rtree_knn(N: int):
    D = 128
    K = 8
    points = random_points(N, D)

    rtree = RTree(D, False)
    rtree.build(points)

    c = random_point(D)
    rtree.knn(c, K, False)


def test_benchmark100(benchmark):
    benchmark(rtree_knn, 100)

def test_benchmark100(benchmark):
    benchmark(rtree_knn, 200)

def test_benchmark400(benchmark):
    benchmark(rtree_knn, 400)

def test_benchmark800(benchmark):
    benchmark(rtree_knn, 800)

def test_benchmark1600(benchmark):
    benchmark(rtree_knn, 1600)

def test_benchmark3200(benchmark):
    benchmark(rtree_knn, 3200)

def test_benchmark6400(benchmark):
    benchmark(rtree_knn, 6400)

def test_benchmark12800(benchmark):
    benchmark(rtree_knn, 12800)

