"""
Base class for model.
"""

import numpy as np 


class Model:
    def __init__(self, *layers) -> None:
        """Create model with set of layers.
        Args:
            layers: list of layers.
        """
        self.layers = list(layers)
        for i in range(len(self.layers) - 1):
            assert self.layers[i].output_size == self.layers[i + 1].input_size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Pass through input."""
        for layer in self.layers:
            x = layer(x)
            #print(x)
        return x
