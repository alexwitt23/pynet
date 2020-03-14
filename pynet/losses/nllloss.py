"""Negative log likelihood loss.

Assumes that input log(p)

loss = -sum(p(target)) / batch_size

p(target) should be 0 if, if prob 1 that input is type target.
"""

import numpy as np 


class NLLLoss:

    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray, target: np.ndarray) -> float:
        print("WWW")
        self.input = x
        self.target = target
        self.batch_size = self.input.shape[0]
        return -1 * np.sum(x[range(self.batch_size), self.target[:, 0]]) / self.batch_size

    def backwards(self) -> np.ndarray:
        self.grad = np.zeros((self.batch_size, self.input.shape[1]))
        self.grad[range(self.batch_size), self.target[:, 0]] = -1
        return self.grad
