from typing import List
import numpy as np

Point = np.ndarray  # d-dimensional vector
Box = np.ndarray  # 2 * d dimensional vector


class Index:
    dim = int
    points: List[Point]

    def __init__(self, dim: int):
        self.dim = dim

    def build(self, points: List[Point]):
        self.points = points.copy()

    def knn(self, center: Point, k: int) -> List[int]:
        ans = list(range(len(self.points)))
        ans.sort(key=lambda ix: np.linalg.norm(self.points[ix] - center))
        return ans[:k]

    def contained(self, center: Point, r: float) -> List[int]:
        return [
            ix for ix, p in enumerate(self.points)
            if np.linalg.norm(p - center) <= r
        ]
