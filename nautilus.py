#!/usr/bin python3
"""
Attempt at training a model on spiral pattern. 
Inspiration from: http://cs231n.github.io/neural-networks-case-study.
"""

import random
import tempfile
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import PIL

import pynet
from pynet.core.model import Model
from pynet.layers.fullyconnected import Linear
from pynet.layers.log_softmax import LogSoftmax
from pynet.losses.nllloss import NLLLoss
from pynet.optimizers.optimizer import sgd


# Setup up the nautilus plot
plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

np.random.seed(0)
N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes
X = np.zeros((N * K, D))
y = np.zeros(N * K, dtype="uint8")
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1, 1])
plt.ylim([-1, 1])

# Create a function that we can use model to predict a plane
def plot_plane(idx: int, model: pynet.core.model.Model, tmp_dir: pathlib.Path):
    """Take in a model and plot the plane. Write the resultant figure
    into a directory."""

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    xs = xx.ravel()
    ys = yy.ravel()
    Z = np.zeros((xs.shape[0]))
    for ii in range(xs.shape[0]):
        temp = model(np.array([[xs[ii], ys[ii]]]))
        Z[ii] = np.argmax(temp, axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    save_path = tmp_dir / f"img_{idx}.png"
    plt.savefig(save_path)
    plt.close()


def save_frames_as_gif(frame_dir: pathlib.Path, save_path: pathlib.Path):
    img, *imgs = [PIL.Image.open(img) for img in sorted(list(frame_dir.glob("*.png")))]
    img.save(
        fp=save_path,
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=100,
        loop=0,
    )


if __name__ == "__main__":

    """Create a model."""
    model = Model(
        Linear(2, 20, bias=True),
        Linear(20, 10, bias=True),
        Linear(10, 3, bias=True, activation=None),
        LogSoftmax(input_size=3, axis=1),
    )

    loss_fn = NLLLoss()
    optimizer = sgd(model, lr=1e-0, weight_decay=1e-3)

    temp_dir = pathlib.Path(tempfile.TemporaryDirectory().name)
    temp_dir.mkdir()
    print(f"Writing frames to {temp_dir}.")
    # Train the model
    for i in range(2000):
        out = model(np.array(X))
        if not isinstance(out, tuple):
            out = (out,)

        out += (np.expand_dims(np.array(y), axis=1),)
        loss = loss_fn(*out)
        optimizer.step(loss_fn.backwards())

        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")

        if i % 50 == 0:
            plot_plane(i, model, temp_dir)

    # evaluate training set accuracy
    scores = model(X)
    predicted_class = np.argmax(scores, axis=1)
    print("training accuracy: %.2f" % (np.mean(predicted_class == y)))

    # Create gif
    save_gif = pathlib.Path("nautilus.gif")
    save_frames_as_gif(temp_dir, save_gif)
