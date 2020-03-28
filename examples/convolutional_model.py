#!/usr/bin/env python3
"""
Usage: PYTHONPATH=$PWD examples/convolutional_model.py
"""
import numpy as np

from pynet.nn import layers, nllloss, optimizer
from pynet.core import model

model = model.Model(
    layers.Conv2D(3, 3, (2, 2), stride=1, padding=0),
    layers.Conv2D(3, 3, (2, 2), stride=1, padding=0),
    layers.Conv2D(3, 3, (2, 2), stride=1, padding=0),
    layers.Conv2D(3, 3, (2, 2), stride=1, padding=0),
    layers.Conv2D(3, 3, (2, 2), stride=1, padding=0),
    layers.Conv2D(3, 3, (2, 2), stride=1, padding=0),
    layers.Flatten(),
    layers.Linear(48, 2),
    layers.LogSoftmax(2, axis=1)
)

loss_fn = nllloss.NLLLoss()
optimizer = optimizer.sgd(model, lr=1.5e-2, momentum=0.9, weight_decay=1e-4, nesterov=True)
    
for epoch in range(30):
    out = model(np.random.randn(1, 10, 10, 3))
    if not isinstance(out, tuple):
        out = (out,)

        out += (np.expand_dims(np.array(1), axis=1),)
        loss = loss_fn(*out)
        optimizer.step(loss_fn.backwards())

        # Zero out the gradient accumulation
        if i % 5 == 0:
            optimizer.zero_grad()
