import unittest
from typing import List
from lib.rtree import RTree
from lib.baseline import Index
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

def secuential_knn(index):
    c = random_point(128)
    index.knn(c, 8)


def test_benchmark100RTree(benchmark):
    N = 100
    points = random_points(N, 128)
    rtree = RTree(128, False)
    rtree.build(points)
    benchmark(rtree_knn, rtree)

def test_benchmark100Secuential(benchmark):
    N = 100
    points = random_points(N, 128)
    index = Index(128)
    index.build(points)
    benchmark(secuential_knn, index)

def test_benchmark200RTree(benchmark):
    N = 200
    points = random_points(N, 128)
    rtree = RTree(128, False)
    rtree.build(points)
    benchmark(rtree_knn, rtree)

def test_benchmark200Secuential(benchmark):
    N = 200
    points = random_points(N, 128)
    index = Index(128)
    index.build(points)
    benchmark(secuential_knn, index)

def test_benchmark400RTree(benchmark):
    N = 400
    points = random_points(N, 128)
    rtree = RTree(128, False)
    rtree.build(points)
    benchmark(rtree_knn, rtree)

def test_benchmark400Secunetial(benchmark):
    N = 400
    points = random_points(N, 128)
    index = Index(128)
    index.build(points)
    benchmark(secuential_knn, index)

def test_benchmark800RTree(benchmark):
    N = 800
    points = random_points(N, 128)
    rtree = RTree(128, False)
    rtree.build(points)
    benchmark(rtree_knn, rtree)

def test_benchmark800Secunetial(benchmark):
    N = 800
    points = random_points(N, 128)
    index = Index(128)
    index.build(points)
    benchmark(secuential_knn, index)

def test_benchmark1600RTree(benchmark):
    N = 1600
    points = random_points(N, 128)
    rtree = RTree(128, False)
    rtree.build(points)
    benchmark(rtree_knn, rtree)

def test_benchmark1600Secunetial(benchmark):
    N = 1600
    points = random_points(N, 128)
    index = Index(128)
    index.build(points)
    benchmark(secuential_knn, index)

def test_benchmark3200RTree(benchmark):
    N = 3200
    points = random_points(N, 128)
    rtree = RTree(128, False)
    rtree.build(points)
    benchmark(rtree_knn, rtree)

def test_benchmark3200Secunetial(benchmark):
    N = 3200
    points = random_points(N, 128)
    index = Index(128)
    index.build(points)
    benchmark(secuential_knn, index)

def test_benchmark6400RTree(benchmark):
    N = 6400
    points = random_points(N, 128)
    rtree = RTree(128, False)
    rtree.build(points)
    benchmark(rtree_knn, rtree)

def test_benchmark6400Secuential(benchmark):
    N = 6400
    points = random_points(N, 128)
    index = Index(128)
    index.build(points)
    benchmark(secuential_knn, index)

def test_benchmark12800RTree(benchmark):
    N = 12800
    points = random_points(N, 128)
    rtree = RTree(128, False)
    rtree.build(points)
    benchmark(rtree_knn, rtree)

def test_benchmark12800Secuential(benchmark):
    N = 12800
    points = random_points(N, 128)
    index = Index(128)
    index.build(points)
    benchmark(secuential_knn, index)
        
