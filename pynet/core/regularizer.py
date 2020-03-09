"""Some simple regularization functions."""

import numpy as np 


class l2:
    def __init__(self, alpha=1e-4) -> None:
        self.alpha = alpha
    
    @staticmethod
    def apply(w: np.ndarray, decay: float) -> np.ndarray:
        """Calculate the l2 normalization penalty during backprop."""
        return decay * w


# TODO l1 regularization
class l1: 
    def __init__(self, alpha: float = 1e-4) -> None:
        self.alpha = alpha


# Dictionary of regularization for layer weights
regularize_dict = {
    "l2": l2,
}