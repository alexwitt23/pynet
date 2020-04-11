#!/usr/bin/env python3
"""Usage: PYTHONPATH=$(pwd) examples/regression_ex.py"""

import numpy as np

from pynet.supervised import regression


if __name__ == "__main__":

    np.random.seed(0)

    from sklearn.datasets.samples_generator import make_blobs
    import matplotlib.pyplot as plt

    num_points = 100
    y = np.sort(np.random.uniform(-1, 1, (num_points)))
    x = np.linspace(-1, 1, num=num_points)

    linear_regression = regression.LinearRegression()

    a, b = linear_regression.fit(x, y)

    # Get the range to plot line on

    plt.scatter(x, y, c="blue", s=50, cmap="viridis")
    plt.plot(x, a * x + b)
    plt.show()
