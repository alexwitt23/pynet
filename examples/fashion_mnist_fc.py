#!/usr/bin/env python3
"""CNN model trained with Fashion-MNIST.

Usage: PYTHONPATH=$(pwd) examples/fashion_mnist_fc.py"""

import pathlib

import numpy as np
from tensorflow import keras

from pynet.nn import layers, activations, losses, optimizer
from pynet.core.model import Model


if __name__ == "__main__":

    """Create a model."""
    model = Model(
        layers.Flatten(),
        layers.Linear(28 * 28, 128, bias=True),
        layers.Linear(128, 10, bias=True),
        layers.LogSoftmax(input_size=10, axis=1),
    )
    loss_fn = losses.NLLLoss()
    optimizer = optimizer.sgd(model, lr=1e-1, momentum=0.9, weight_decay=1e-4, nesterov=True)

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    batch_size = 6000
    # Train the model
    for i in range(100000):
        for b in range(0, train_images.shape[0], batch_size):
            out = model(train_images[i:batch_size, :, :] / 255)
            if not isinstance(out, tuple):
                out = (out,)
            out += (np.expand_dims(train_labels[i:batch_size], axis=1),)
            loss = loss_fn(*out)
            optimizer.step(loss_fn.backwards())

            optimizer.zero_grad()

            if b % 6000 == 0:
                print(f"Epoch {i}, Loss: {loss}")

        # Loop over eval data
        num_right = 0
        for b in range(0, test_images.shape[0], batch_size):
            out = model(test_images[i:batch_size, :, :] / 255)

            if b % 6000 == 0:
                print(f"Epoch {i}, Loss: {loss}")
            num_right += (np.argmax(out, axis=1) == test_labels[i:batch_size]).sum()
        
        print(f"Eval accuracy: {num_right / test_images.shape[0]}")
