"""Collection of activations."""

__author__ = "Alex Witt <awitt2399@utexas.edu>"

import numpy as np

from pynet.nn.layers import Layer


class ReLU(Layer):
    """Rectified Linear Units:

    y = ReLU(x) = { x if x > 0; 0 if x <= 0}
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backwards(self, dy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Takes in the dL/dy and y and applies relu backprop.
        dy = d/dx(ReLU(x)) = { 1 if x > 0; 0 if x <=0 }
        """
        dy[y <= 0] = 0
        return dy

    def parameters(self) -> int:
        return 0

    def input_size(self) -> np.ndarray:
        None

    def output_size(self) -> np.ndarray::
        None


class LeakyReLU:
    """Leaky Rectified Linear Units:

    - An attempt to combate the "dying ReLU" problem, or when the 
    ReLU gradient vanish. 

    y = ReLU(x) = { x if x > 0; alpha if x <= 0}
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = -alpha

    @staticmethod
    def apply(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @classmethod
    def backprop(self, dy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Takes in the dL/dy and y and applies relu backprop.
        dy = d/dx(ReLU(x)) = { 1 if x > 0; 0 if x <=0 }
        """
        dy[y <= 0] = self.alpha
        return dy


activations_dict = {
    "ReLU": ReLU,
    "LeakyReLU": LeakyReLU,
}
