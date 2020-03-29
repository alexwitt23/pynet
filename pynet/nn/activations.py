"""Collection of activation functions."""

__author__ = "Alex Witt <awitt2399@utexas.edu>"

import numpy as np

from pynet.nn.layers import Layer


class ReLU(Layer):
    """Rectified Linear Units:

    y = ReLU(x) = { x if x > 0; 0 if x <= 0}
    """

    def __init__(self) -> None:
        super().__init__()
        self.weights = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backwards(self, dy: np.ndarray) -> np.ndarray:
        """Takes in the dL/dy and y and applies relu backprop.
        dy = d/dx(ReLU(x)) = { 1 if x > 0; 0 if x <=0 }
        """
        dy[self.input <= 0] = 0
        return dy, None

    def update(self, grad: np.ndarray) -> None:
        pass

    def parameters(self) -> int:
        return 0

    def input_size(self) -> int:
        return 0

    def output_size(self) -> int:
        return 0


class ReLU6(Layer):
    """ReLU6 based on this paper: 
    http://www.cs.utoronto.ca/~kriz/conv-cifar10-aug2010.pdf which 
    recommends capping activations at 6 to force model to learn sparse features."""

    def __init__(self) -> None:
        super().__init__()
        self.weights = False

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.minimum(np.maximum(0, x), 6)

    def backwards(self, dy: np.ndarray) -> np.ndarray:
        """Takes in the dL/dy and y and applies relu backprop.
        dy = d/dx(ReLU(x)) = { 1 if x > 0; 0 if x <=0 }
        """
        dy[self.input <= 0] = 0
        return dy

    def update(self, grad: np.ndarray) -> None:
        pass

    def parameters(self) -> int:
        return 0

    def input_size(self) -> int:
        return 0

    def output_size(self) -> int:
        return 0


class LeakyReLU:
    """Leaky Rectified Linear Units:

    - An attempt to combate the "dying ReLU" problem, or when the 
    ReLU gradient vanish. 

    y = ReLU(x) = { x if x > 0; alpha if x <= 0}
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = -alpha

    def apply(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backprop(self, dy: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Takes in the dL/dy and y and applies relu backprop.
        dy = d/dx(ReLU(x)) = { 1 if x > 0; 0 if x <=0 }
        """
        dy[y <= 0] = self.alpha
        return dy
