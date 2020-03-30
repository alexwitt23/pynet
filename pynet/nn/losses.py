"""Collection of loss functions."""

import abc

import numpy as np


class Loss:
    @abc.abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Must override __call__ method!")

    @abc.abstractmethod
    def backwards(self, dout: np.ndarray) -> np.ndarray:
        """Send the gradient through this layer and update weights."""
        raise NotImplementedError("Must override backprop method!")


class NLLLoss(Loss):
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
        return -1 * np.mean(x[range(self.batch_size), self.target[:, 0]])

    def backwards(self) -> np.ndarray:
        self.grad = np.zeros((self.batch_size, self.input.shape[1]))
        self.grad[range(self.batch_size), self.target[:, 0]] = -1
        return self.grad


class MSE(Loss):
    """Simple mean squared error loss."""

    def __init__(self, input_size) -> None:
        self.input_size = input_size

    def __call__(self, pred, target) -> np.float:
        self.target = target
        self.pred = pred
        self.loss = 0.5 * np.mean(np.square(target - pred)) / self.target.shape[1]
        return self.loss

    def backwards(self) -> np.ndarray:
        # Derivatives
        grad = (self.target - self.pred) / self.target.shape[1]
        return grad


# TODO
class CrossEntropyLoss(Loss):
    """Cross Entropy Loss.
    Based on Kullback-Leibler (KL) Divergence Dkl(P||Q) which can also
    be written as Expectation of log(P(x) / Q(x)) given P(x). KL Divergence is 
    0 only is the output prob density Q(x) is equal to the know P(x). 
    Note, KL Divergence is asymmetric p72.

    Cross entopy is also based on Shannon entropy where **self_information**
    is defined as I(x) = -log(P(x)). This then leads to the uncertaintiy in an entire 
    distribution p71. 

    If P(x) = 1, then log(P(x)) = 0, meaning it is certain to happen. A loss function
    won't penalize for this correct prediction.

    NOTE: Usually softmax and cross entropy are combined into one step because the 
    their combined chain rule works out cleanly.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, x: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Get the log loss of the target vs predicted. 
        
        Args:
            x: predictions must be in form of distribution.
            target: correct class.
            
        Returns:
            Log likelihood loss calculation.
        """
        self.pred = x
        self.target = target

        n = x.shape[0]  # Batch size
        log_likelihood = -np.log(x[n, target])
        return np.average(log_likelihood)

    def backwards(self) -> np.ndarray:
        return np.divide(self.target, self.pred)
