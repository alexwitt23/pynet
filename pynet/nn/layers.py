"""Contains all the layers in this library."""

from typing import Tuple
import abc

import numpy as np

import pynet


class Layer(abc.ABC):
    """Define a collection of member functions that must be 
    implemented by all layers."""

    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Must override __call__ method!")

    @abc.abstractmethod
    def backwards(self, dout: np.ndarray) -> np.ndarray:
        """Send the gradient through this layer and update weights."""
        raise NotImplementedError("Must override backprop method!")

    @abc.abstractmethod
    def parameters(self) -> int:
        """Number of trainable params in the model."""
        raise NotImplementedError("Must implement the number of layer parameters!")

    @abc.abstractmethod
    def input_size(self) -> int:
        """Size of layer's input size."""
        raise NotImplementedError("Implement input_size!")

    @abc.abstractmethod
    def output_size(self) -> int:
        """Size of layer's output size."""
        raise NotImplementedError("Implement output_size!")


class Linear(Layer):
    """Fully connected layer defined by y = Wx + b."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False) -> None:
        """Linear, or fully connected, layer.
        
        Args:
            input_size: number of input units. [N x M]
            output_size: number of output units. [N X D]
            bias: To include the bias vector or not.

        Returns:
            Matrix of size [N x D]
        """
        super().__init__()
        self.input_dim = input_size
        self.output_dim = output_size
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.num_params = input_size * output_size

        self.use_bias = bias

        if self.use_bias:
            self.biases = np.zeros((1, output_size))
        else:
            self.biases = None
        
        self.optim_weights: pynet.nn.optimizer.Optimizer = None
        self.optim_biases: pynet.nn.optimizer.Optimizer = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass.
        
        Args:
            - x: Input Tensor.

        Returns:
            Apply an affine transformation y = Wx + b.
        """
        self.input = x
        self.out = np.dot(self.input, self.weights)
        # Apply bias if included
        if self.use_bias:
            self.out += self.biases

        return self.out

    def backwards(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backprop of gradient to weights, biases, and chain rule.
        See derivation: http://cs231n.stanford.edu/handouts/linear-backprop.pdf.

        Args:
            dout: The derivative of loss w.r.t this layer's output. (N, M)
        
        Returns:
            Derivative of loss w.r.t this layer's input.
        """
        # Calc grad w.r.t input
        din = np.dot(dout, self.weights.transpose())
        # Calc grad w.r.t weights
        self.dw = np.dot(self.input.transpose(), dout)
        # Now call the optimizer to update this layer's weights and biases
        self.optim_weights.update(self.weights, self.dw)

        if self.biases is not None:
            self.optim_biases.update(self.biases, dout.sum(axis=0))
            
        return din

    def parameters(self) -> int:
        return self.input_dim * self.output_dim

    def input_size(self) -> int:
        return self.input_dim

    def output_size(self) -> int:
        return self.output_dim


class Conv2D(Layer):
    """Two dimensional convolutional layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = in_channels
        self.num_filters = out_channels
        self.weights = True
        self.kernel_size = kernel_size  # (W, H)
        self.stride = stride
        self.padding = padding

        # Create numpy filter volume
        self.kernel = np.random.randn(kernel_size[0], kernel_size[1], out_channels)

        if bias:
            self.biases = np.zeros((1, out_channels))
        else:
            self.biases = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through convolutional layer. This will be implemented in a 
        slightly confusing fashion, but the idea is to level numpy's matrix math to 
        speed up the calculations."""

        assert len(x.shape) == 4, "Input must be [N, W, H, C]"
        # Get the width and height of the incoming data.
        batch_size, width, height, filters_in = x.shape
        assert filters_in == self.input_size, f"Improper input channles: {x.shape}!"
        # Calculate output filter W, H
        width_out = int((width + self.padding - self.kernel_size[0]) / self.stride + 1)
        height_out = int(
            (height + self.padding - self.kernel_size[1]) / self.stride + 1
        )

        # Equivalent to [x for _ in range(kernel_height) for x in range(kernel_width)]
        i = np.repeat(
            np.tile(np.arange(self.kernel_size[0]), self.kernel_size[1]),
            self.input_size,
        )
        # Equivalent to [y for y in range(kernel_height) for _ in range(kernel_width)]
        j = np.tile(
            np.arange(self.kernel_size[1]), self.kernel_size[0] * self.input_size
        )

        # These are the indices of the output width [x for x in range(width_out)]
        # np.repeat turns this in to [x for _ in range(out_height) for x in range(width_out)]
        i1 = self.stride * np.repeat(np.arange(width_out), height_out)

        # This is equivalent to [y for y in range(out_height) for _ in range(width_out)]
        j1 = self.stride * np.tile(np.arange(height_out), width_out)

        i = i.reshape(-1, 1) + i1.reshape(1, -1)
        j = j.reshape(-1, 1) + j1.reshape(1, -1)
        # Get indices for each of the incoming filter repeated by the number of weights one of this
        # layer's filters.
        filter_ids = np.repeat(
            np.arange(self.input_size), self.kernel_size[0] * self.kernel_size[1]
        ).reshape(-1, 1)

        self.img_slices = (
            x[:, i, j, filter_ids]
            .transpose(1, 2, 0)
            .reshape(self.kernel_size[0] * self.kernel_size[1] * self.input_size, -1)
        )
        # Flatten out each filter into a column
        self.weights_col = np.repeat(
            self.kernel.reshape((self.num_filters, -1)), self.input_size, axis=1
        )
        retval = np.dot(
            self.weights_col, self.img_slices
        )  # (num_filters_out, out_w * out_h * batch_size)
        retval = retval.reshape(((self.num_filters, width_out, height_out, batch_size)))

        return retval.transpose(3, 1, 2, 0)

    def backwards(self, dout: np.ndarray) -> np.ndarray:
        # Take the input gradient and reshape for reshaped weights.

        dout = dout.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)

        # Note the similarity to the dense layer:
        return np.dot(dout.transpose(), self.weights_col)

    def input_size(self):
        pass

    def parameters(self) -> int:
        return self.output_size * self.kernel_size[0] * self.kernel_size[1]

    def output_size(self) -> int:
        return self.num_filters

    def update(self, grad: np.ndarray) -> None:
        pass


class LogSoftmax(Layer):
    """Apply log to softmax for more preferable numerical properties."""

    def __init__(self, input_size, axis) -> None:
        super().__init__()
        self.input_dim = input_size
        self.output_dim = input_size
        self.axis = axis
        self.num_params = 0
        self.weights = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        inter = np.exp(x)  # - np.amax(x, axis=self.axis, keepdims=True)
        self.out = np.log(inter) - np.log(inter.sum(axis=self.axis, keepdims=True))
        self.softmax = np.exp(self.out)

        return self.out

    def backwards(self, dout: np.ndarray) -> np.ndarray:
        # Apply gradient, which is 1 - p(x) where x = target.
        # Then complete chain rule with incoming gradient
        self.softmax += dout
        return self.softmax / self.input.shape[0]

    def parameters(self) -> int:
        return self.num_params

    def input_size(self) -> int:
        return self.input_dim

    def output_size(self) -> int:
        return self.output_dim


class Flatten(Layer):
    """Layer that will flatten input."""

    def __init__(self) -> None:
        self.weights = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Assume batch is first..
        x.shape = [batch, _]."""
        self.input_size = x.shape
        return x.reshape(x.shape[0], -1)

    def backwards(self, dout: np.ndarray) -> None:
        # Reshape gradient to input size
        return dout.reshape(self.input_size), None

    def update(self, grad: np.ndarray) -> None:
        pass

    def parameters(self) -> int:
        return 0

    def input_size(self) -> int:
        return 0

    def output_size(self) -> int:
        return 0


class Dropout(Layer):
    """Layer which will randomly cancel out the outputs from units in 
    previous layer."""

    def __init__(self, prob: float = .1) -> None:   
        self.prob = prob
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.mask = np.where(np.random.uniform(size=x.shape) > self.prob, 1, 0)
        return x * self.mask

    def backwards(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask

    def parameters(self) -> int:
        return 0

    def input_size(self) -> int:
        return 0

    def output_size(self) -> int:
        return 0
