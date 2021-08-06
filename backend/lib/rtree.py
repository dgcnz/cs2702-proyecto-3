from rtree import index
from typing import List
import numpy as np

Point = np.ndarray  # d-dimensional vector
Box = np.ndarray  # 2 * d dimensional vector


def to_point(box: Box) -> Point:
    return box[:int(len(box) / 2)]


def to_box(point: Point) -> Box:
    return np.array(list(point) + list(point))


class RTree:
    dim: int
    idx: index.Index

    def __init__(self, dim: int, disk=False):
        p = index.Property()
        p.dimension = dim
        self.dim = dim
        if disk:
            self.idx = index.Index('rtree', interleaved=True, properties=p)
        else:
            self.idx = index.Index(interleaved=True, properties=p)

    def build(self, points=List[Point]):
        for ix, p in enumerate(points):
            assert (len(p) == self.dim)
            self.idx.insert(ix, to_box(p))

    def contained(self, center: Point, r: float) -> List[int]:
        assert (len(center) == self.dim)
        assert (r >= 0)
        box = [x - r for x in center] + [x + r for x in center]
        return [
            item.id for item in self.idx.intersection(box, objects=True)
            if np.linalg.norm(to_point(item.bbox) - center) <= r
        ]

    def knn(self, p: Point, k: int, strict=True) -> List[int]:
        assert (len(p) == self.dim)
        assert (k >= 0)
        ans = list(self.idx.nearest(to_box(p), k))
        if strict:
            return ans[:k]
        return ans

    def get(self, ix: int) -> str:
        with open('answers.txt', 'r') as f:
            paths = str(f.read()).splitlines()
            return paths[ix]
