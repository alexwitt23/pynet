"""
This script might serve as the implementation of the various 
networks constructed in this repo.
"""

import numpy as np 

from layers.fullyconnected import linear
from layers.softmax import softmax
from core.model import model
from losses.mse import mse


if __name__ == '__main__':

    test_model = model(
        linear(input_size=30, output_size=20, bias=False),
        linear(input_size=20, output_size=15, bias=False),
        linear(input_size=15, output_size=10, bias=False),
        linear(input_size=10, output_size=2, bias=False)
    )
    # Loss of choice
    loss_fn = mse(input_size=2)

    # Training on Gaussian distribution
    mu, sigma = 0, 0.1
    target = (np.array([[mu, sigma]]))

    for i in range(5):
        data = np.expand_dims(np.random.normal(mu, sigma, 30), axis=0)
        out = test_model(data)

        if not isinstance(out, tuple):
            out = (out,)

        out += (target,)
        loss = loss_fn(*out)
        test_model.backwards(loss_fn.backprop())
        
        print(f"Loss: {loss}.")