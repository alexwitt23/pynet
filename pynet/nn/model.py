"""Base class for model."""

import pathlib

import numpy as np
import pickle

import pynet


class Model:
    def __init__(self, *layers) -> None:
        """Create model with set of layers.
        Args:
            layers: list of layers.
        """
        self.layers = list(layers)
        # TODO(alex) re implement this
        # for i in range(len(self.layers) - 1):
        #    assert self.layers[i].output_size == self.layers[i + 1].input_size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Pass through input."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> str:
        """Tally up the total number of params."""
        num_params: int = 0
        for layer in self.layers:
            num_params += layer.num_params

        return f"Model contains {num_params} parameters."

    def save(self, path: pathlib.Path) -> None:
        """Save model."""
        with path.open("wb") as f:
            layers_dict = {}
            for idx, layer in enumerate(self.layers):
                layers_dict[idx] = layer.weights
            pickle.dump(layers_dict, f)

        return None

    def load(self, path: pathlib) -> None:
        "Load params."
        print("Loading")
        assert path.exists(), f"{path} does not exist."

        with path.open("rb") as f:
            model_data = pickle.load(f)

            for idx, layer in enumerate(self.layers):
                layer.weights = np.array(model_data[idx])

        return None
