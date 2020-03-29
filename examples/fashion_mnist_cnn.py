#!/usr/bin/env python3
"""CNN model trained with Fashion-MNIST."""

import pathlib

import numpy as np

import pynet
from pynet.core.model import Model
from pynet.layers.convolution2D import Conv2D
from pynet.layers.log_softmax import LogSoftmax
from pynet.losses.nllloss import NLLLoss
from pynet.optimizers.optimizer import sgd

DATA_DIR = pathlib.Path("~/fashion-dataset").expanduser()


# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz


if __name__ == "__main__":

    """Create a model."""
    model = Model(
        Conv2D(in_channels=3, out_channels=4, kernel_size=3, padding=2),
        Conv2D(in_channels=4, out_channels=3, kernel_size=4),
    )

    input_img = np.array([np.ones((64, 64, 3))])
    out = model(input_img)
