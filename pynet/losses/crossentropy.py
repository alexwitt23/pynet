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

import numpy as np


class CrossEntropyLoss:
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

    def backprop(self) -> np.ndarray:
        return np.divide(self.target, self.pred)
