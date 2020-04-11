#!/usr/bin/env python3
"""CNN model trained with Fashion-MNIST."""

import pathlib

import numpy as np
from tensorflow import keras

from pynet.nn import model, layers, optimizer, losses

# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
# http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz


if __name__ == "__main__":

    """Create a model."""
    cnn_model = model.Model(
        layers.Conv2D(in_channels=1, out_channels=32, kernel_size=(3, 3)),
        layers.Conv2D(in_channels=32, out_channels=32, kernel_size=(3, 3)),
        layers.Conv2D(in_channels=32, out_channels=2, kernel_size=(3, 3)),
        layers.Flatten(),
        layers.Linear(968, 10, bias=True),
        layers.LogSoftmax(input_size=10, axis=1)
    )

    loss_fn = losses.NLLLoss()
    optimizer = optimizer.SGD(
        cnn_model, lr=2e-1, momentum=0.9, weight_decay=1e-4, nesterov=False
    )

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    batch_size = 6000
    # Train the model
    for epoch in range(100000):
        for b in range(0, train_images.shape[0], batch_size):
            X = train_images[b : b + batch_size, :, :] / 255
            X = np.expand_dims(X, axis=3)

            out = cnn_model(X)
            if not isinstance(out, tuple):
                out = (out,)
            out += (np.expand_dims(train_labels[b : b + batch_size], axis=1),)
            loss = loss_fn(*out)
            optimizer.step(loss_fn.backwards())

            if b % 6000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        if epoch % 10 == 0:
            optimizer.lr /= 10
