"""ReLU Activation."""

import numpy as np 


class ReLU:
    """Rectified Linear Units:

    y = ReLU(x) = { x if x > 0; 0 if x <= 0}
    """
    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def backprop(dy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Takes in the dL/dy and y and applies relu backprop.
        dy = d/dx(ReLU(x)) = { 1 if x > 0; 0 if x <=0 }
        """
        dy[y <= 0] = 0
        return dy