"""Two dimensional convolutional layer."""

__author__ = "Alex Witt <awitt2399@utexas.edu>"

import numpy as np


class Conv2D:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        self.input_size = in_channels
        self.output_size = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = bias

        # Create numpy filter volume
        self.kernel = np.random.randn(kernel_size, kernel_size, out_channels)

        if self.use_bias:
            self.biases = np.zeros((1, out_channels))
        else:
            self.biases = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Pad the input
        self.input = np.pad(x, (self.padding,))
        print(self.input.shape)
        # Don't know HW of input until here..
        self.output_shape = int(
            (1 / self.stride) * (2 * self.padding + x.shape[1] - self.kernel_size) + 1
        )
        # NWHC
        self.output = np.zeros(
            (x.shape[0], self.output_shape, self.output_shape, self.output_size)
        )
        assert len(x.shape) == 4
        # Loop over the batch
        for item in range(x.shape[0]):
            # Loop over the filters
            for f in range(self.output_size):
                # Apply fiter to entire volume and loop over the input
                for j in range(0, x.shape[1] - self.kernel_size, self.stride):
                    for k in range(0, x.shape[2] - self.kernel_size, self.stride):
                        # Multiply input by filter
                        self.output[item, j, k, f] = np.sum(
                            np.multiply(
                                x[item, j + self.kernel_size, k + self.kernel_size, :],
                                self.kernel[:, :, f],
                            )
                        )

                if self.use_bias:
                    self.output += self.biases[0, f]

        return self.output
