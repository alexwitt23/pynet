#!/usr/bin/env python3
"""CNN model trained with Fashion-MNIST. We'll use TF for loading the data.

Usage: PYTHONPATH=$(pwd) examples/fashion_mnist_fc.py"""

import pathlib

import numpy as np
from tensorflow import keras

from pynet.nn import layers, activations, losses, optimizer, model


if __name__ == "__main__":

    """Create a model."""
    fc_model = model.Model(
        layers.Flatten(),
        layers.Linear(28 * 28, 300, bias=True),
        layers.BatchNorm(300),
        activations.ReLU6(),
        layers.Linear(300, 10, bias=True),
        layers.LogSoftmax(input_size=10, axis=1),
    )
    loss_fn = losses.NLLLoss()
    optimizer = optimizer.sgd(
        fc_model, lr=2e-1, momentum=0.9, weight_decay=1e-4, nesterov=False
    )

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    batch_size = 6000
    # Train the model
    for epoch in range(100000):
        for b in range(0, train_images.shape[0], batch_size):

            out = fc_model(train_images[b : b + batch_size, :, :] / 255)
            if not isinstance(out, tuple):
                out = (out,)
            out += (np.expand_dims(train_labels[b : b + batch_size], axis=1),)
            loss = loss_fn(*out)
            optimizer.step(loss_fn.backwards())

            if b % 6000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        if epoch % 10 == 0:
            optimizer.lr /= 10

        # Loop over eval data
        num_right = 0
        for b in range(0, test_images.shape[0], batch_size):
            out = fc_model(test_images[b : b + batch_size, :, :] / 255)
            num_right += (
                np.argmax(out, axis=1) == test_labels[b : b + batch_size]
            ).sum()

        print(f"Eval accuracy: {num_right / test_images.shape[0]}")
