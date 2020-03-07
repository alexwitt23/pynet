"""Some simple regularization functions."""

import numpy as np 


class l2:
    def __init__(self, alpha=1e-4) -> None:
        self.alpha = alpha
    
    def __call__(self, x) -> np.ndarray:
        return np