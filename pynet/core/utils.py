import numpy as np


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """Calculate the euclidean distance between two vectors with D
    dimensions."""
    assert len(x1.shape) == 1 and len(x2.shape) == 1
    return np.sqrt(np.sum((p1 - p2) ** 2 for p1, p2 in zip(x1, x2)))
