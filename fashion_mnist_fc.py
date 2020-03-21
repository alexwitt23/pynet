#!/usr/bin/env python3
"""CNN model trained with Fashion-MNIST."""

import pathlib 
import numpy as np 
from tensorflow import keras

import pynet
from pynet.core.model import Model
from pynet.layers.fullyconnected import Linear
from pynet.layers.flatten import Flatten
from pynet.layers.log_softmax import LogSoftmax
from pynet.losses.nllloss import NLLLoss
from pynet.optimizers.optimizer import sgd


if __name__ == "__main__":

    """Create a model."""
    model = Model(
        Flatten(),
        Linear(28*28, 128, bias=True),
        Linear(128, 10, bias=True, activation=None),
        LogSoftmax(input_size=10, axis=1),
    )
    loss_fn = NLLLoss()
    optimizer = sgd(model, lr=1e-5, momentum=0.9, weight_decay=1e-4, nesterov=True)

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    batch_size = 100
    # Train the model
    for i in range(80000):
        for b in range(0, train_images.shape[0], batch_size):
            out = model(train_images[i : batch_size, :, :])
            if not isinstance(out, tuple):
                out = (out,)
            out += (np.expand_dims(train_labels[i : batch_size], axis=1),)
            loss = loss_fn(*out)
            optimizer.step(loss_fn.backwards())

            # Zero out the gradient accumulation
            if i % 100 == 0:
                optimizer.zero_grad()

            if b % 6000 == 0:
                print(f"Iteration {i}, Loss: {loss}")
