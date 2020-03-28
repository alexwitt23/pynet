
import numpy as np 

class NLLLoss:
    """Negative log likelihood loss.

    Assumes that input log(p)

    loss = -sum(p(target)) / batch_size

    p(target) should be 0 if, if prob 1 that input is type target."""

    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray, target: np.ndarray) -> float:
        self.input = x
        self.target = target
        self.batch_size = self.input.shape[0]
        return -1 * np.mean(x[range(self.batch_size), self.target[:]])

    def backwards(self) -> np.ndarray:
        self.grad = np.zeros((self.batch_size, self.input.shape[1]))
        self.grad[range(self.batch_size), self.target[:]] = -1
        return self.grad
