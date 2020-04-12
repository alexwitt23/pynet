#!/usr/bin/env python3
"""Implementation of different types of regression.

Source:
https://towardsdatascience.com/machine-learning-for-beginners-d247a9420dab
"""
from typing import List

import numpy as np


class LinearRegression:
    # TODO(alex) implement with gradient?
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


# TODO
class MultipleLinearRegression:
    def __init__(self) -> None:
        pass

    def fit(self, x: np.ndarray, y: np.ndarray):
        """ An analytical solution exists. We can also
        solve iteratively for practice. """

        # Do the analytical solution
        # y = Beta * X is the original equation,
        # solving for Beta: Beta = (X * X').inv * X * y.


# TODO(alex) Polynomial Regression
class PolynomialRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray, degree: int = 1) -> None:
        self.x = x
        self.y = y
        self.beta = np.random.uniform(size=(degree + 1))
        self.degree = degree

    def fit(self) -> np.ndarray:

        for _ in range(100000):
            out = np.polynomial.polynomial.polyval(self.x, self.beta)
            loss = np.mean((self.y - out) ** 2)
            print(loss)

            # dL/dy 
            dL_dy = self.y - out

            dbeta = (
                np.dot(
                    np.array([(n + 1) * (self.x ** n) for n in range(self.degree + 1)]),
                    dL_dy
                )
                / self.x.shape[0]
            )
            self.beta += 0.0000000000001 * dbeta

        return self.beta


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
