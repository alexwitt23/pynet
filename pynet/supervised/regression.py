#!/usr/bin/env python3
""" Implementation of different types of regression.

Source: https://towardsdatascience.com/machine-learning-for-beginners-d247a9420dab """

from typing import List

import numpy as np


class LinearRegression:

    """ Linear regression with least squares error. This is a really succinct derivation
    using calculus: https://www.youtube.com/watch?v=ewnc1cXJmGA. """

    def fit(self, x: np.ndarray, y: np.ndarray) -> List[float]:
        """ Function which will perform the linear regression.

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


class MultipleLinearRegression:
    """ Multiple linear regression describes how one variable is influenced by more than
    one input variables. The derivation is detailed here: 
    http://pillowlab.princeton.edu/teaching/mathtools16/slides/lec10_LeastSquaresRegression.pdf.

    The idea is to have **B** represent the coefficients of your [x_0, ..., x_i]
    independent variable vector. Therefore you want to minimize the mean squared error
    between Bx and Y, or e^2 = ||Y - Bx||^2.
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Function which will perform the multiple linear regression.

        Args:
            x: input independent variable data.
            y: dependent variable.

        """
        return np.linalg.inv(x.transpose() * x) * x.transpose() * y



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
