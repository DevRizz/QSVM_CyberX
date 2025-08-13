from collections import deque
from typing import Optional, Tuple
import numpy as np

class SlidingBuffer:
    def __init__(self, maxlen: int = 512):
        self.maxlen = maxlen
        self.X = deque(maxlen=maxlen)
        self.y = deque(maxlen=maxlen)

    def add(self, x: np.ndarray, y: Optional[int] = None):
        self.X.append(x)
        self.y.append(y if y is not None else -1)

    def labeled(self) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for xi, yi in zip(self.X, self.y):
            if yi in (0, 1):
                xs.append(xi)
                ys.append(yi)
        if xs:
            return np.array(xs), np.array(ys)
        return np.empty((0,)), np.empty((0,))

    def all(self) -> Tuple[np.ndarray, np.ndarray]:
        if len(self.X) == 0:
            return np.empty((0,)), np.empty((0,))
        return np.array(self.X), np.array(self.y)
