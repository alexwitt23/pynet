import numpy as np


class NLLLoss:
    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray, target: np.ndarray) -> float:
        self.input = x
        self.target = target
        self.batch_size = self.input.shape[0]
        print(x.shape, self.target.shape)
        return -1 * np.sum(x[range(self.batch_size), self.target[:]]) / self.batch_size

    def backwards(self) -> np.ndarray:
        self.grad = np.zeros((self.batch_size, self.input.shape[1]))
        self.grad[range(self.batch_size), self.target[:]] = -1
        return self.grad
