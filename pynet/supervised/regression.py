#!/usr/bin/env python3
"""Implementation of different types of regression.

Source:
https://towardsdatascience.com/machine-learning-for-beginners-d247a9420dab
"""
from typing import List

import numpy as np


class LinearRegression:
    # TODO(alex) implement with gradient
    """Linear regression with least squares error.
    This is a really succinct derivation using calculus:
    https://www.youtube.com/watch?v=ewnc1cXJmGA."""

    def __init__(self) -> None:
        pass

    def fit(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        """Fitting function.

        Args:
            x: input independent variable data.
            y: dependent variable.

        Return:
            [a, b] correponding to y = ax + b.
        """
        assert len(x.shape) == 1, "Independent variable must be one dimension!"
        assert len(y.shape) == 1, "Dependent variable must be one dimension!"

        x_avg = np.mean(x)
        y_avg = np.mean(y)

        xy_covariance = np.sum(np.dot(x - x_avg, y - y_avg))
        x_variance = np.sum(np.square(x - x_avg))

        a = xy_covariance / x_variance
        b = y_avg - a * x_avg

        return [a, b]


# TODO(alex) Multiple Linear Regression
#
# TODO(alex) Polynomial Regression
#
# TODO(alex) Support Vector Machine
#
# TODO(alex) Ridge Regression
#
# TODO(alex) Lasso Regression
#
# TODO(alex) Elastic Net
#
# TODO(alex) Baysian Regression
#
# TODO(alex) Decision Tree Regression
#
# TODO(alex) Random Forest Regression
