#!/usr/bin/env python3
"""

Usage: PYTHONPATH=$PWD examples/convolutional_model.py
"""
import numpy as np

from pynet.nn import layers
from pynet.core import model

model = model.Model(layers.Conv2D(1, 3, (2, 2), stride=1, padding=0))

print(model(np.ones((2, 10, 10, 1))).shape)
