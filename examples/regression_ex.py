#!/usr/bin/env python3
"""Usage: PYTHONPATH=$(pwd) examples/regression_ex.py"""

import numpy as np

from pynet.supervised import regression


if __name__ == "__main__":

    np.random.seed(0)

    from sklearn.datasets.samples_generator import make_blobs
    import matplotlib.pyplot as plt

    """
    num_points = 100
    y = np.sort(np.random.uniform(-1, 1, (num_points)))
    x = np.linspace(-1, 1, num=num_points)

    linear_regression = regression.LinearRegression()

    a, b = linear_regression.fit(x, y)

    # Get the range to plot line on
    plt.scatter(x, y, c="blue", s=50, cmap="viridis")
    plt.plot(x, a * x + b)
    plt.show()
    """
    np.random.seed(0)
    x = np.sort(np.random.normal(-10, 10, (200)))
    y = np.polynomial.polynomial.polyval(x, [6, -1, 1, -2])
    plt.scatter(x, y, s=10)

    mlr = regression.PolynomialRegression(x, y, degree=3)
    beta = mlr.fit()

    out = np.polynomial.polynomial.polyval(x, beta)
    plt.plot(x, out)
    plt.show()
