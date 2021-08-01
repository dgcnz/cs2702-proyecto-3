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


class TestRTree(unittest.TestCase):
    def test_knn(self):
        N = 10
        D = 3
        T = 1000
        points = random_points(N, D)

        baseline = Index(D)
        baseline.build(points)

        rtree = RTree(D, False)
        rtree.build(points)

        for _ in range(T):
            c = random_point(D)
            k = randint(0, N)
            bans = set(baseline.knn(c, k))
            rans = set(rtree.knn(c, k, False))
            self.assertTrue(bans.issubset(rans))

    def test_contained(self):
        N = 10
        D = 3
        T = 1000
        points = random_points(N, D)

        baseline = Index(D)
        baseline.build(points)

        rtree = RTree(D, False)
        rtree.build(points)

        for _ in range(T):
            c = random_point(D)
            r = randint(0, 1000)
            bans = set(baseline.contained(c, r))
            rans = set(rtree.contained(c, r))
            self.assertEqual(bans, rans)


if __name__ == '__main__':
    unittest.main()
