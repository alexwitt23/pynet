#!/usr/bin/env python3
"""
Usage: PYTHONPATH=$PWD examples/convolutional_model.py
"""
import numpy as np

from pynet.nn import layers
from pynet.core import model

model = model.Model(
    layers.Conv2D(3, 3, (1, 1), stride=1, padding=0),
    layers.Conv2D(3, 3, (2, 2), stride=1, padding=0),
    layers.Conv2D(3, 3, (2, 2), stride=1, padding=0),
    layers.Conv2D(3, 3, (2, 2), stride=1, padding=0),
    layers.Conv2D(3, 3, (2, 2), stride=1, padding=0),
    layers.Conv2D(3, 3, (2, 2), stride=1, padding=0),
)

print(model(np.ones((1, 10, 10, 3))).shape)
