"""
Building on guassian1 by adding activations
"""

import numpy as np 
import matplotlib.pyplot as plt

from layers.fullyconnected import linear
from layers.softmax import softmax
from core.model import model
from losses.mse import mse


if __name__ == '__main__':

    test_model = model(
        linear(50, 40, bias=True),
        linear(40, 30, bias=True),
        linear(30, 20, bias=True),
        linear(20, 15, bias=True),
        linear(15, 10, bias=True),
        linear(10, 2, bias=True)
    )
    # Loss of choice
    loss_fn = mse(input_size=2)

    # Training on Gaussian distribution
    mu, sigma = 0, 1
    target = (np.array([[mu, sigma]]))

    losses: float = []
    for i in range(100000):
        data = np.expand_dims(np.random.normal(mu, sigma, 50), axis=0)
        out = test_model(data)

        if not isinstance(out, tuple):
            out = (out,)

        out += (target,)
        loss = loss_fn(*out)
        losses.append(loss)
        test_model.backwards(loss_fn.backprop())
        
        print(f"Loss: {loss}.")

    # Final print out
    data = np.expand_dims(np.random.normal(mu, sigma, 50), axis=0)
    print(test_model(data))

    plt.plot(losses)
    plt.show()