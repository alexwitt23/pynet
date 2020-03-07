"""
Sigmoidal activation. This method ensures the gradient is strong whenever 
the answer is incorrect.

Definition: y = sigmoid(w*h + b) 

There are two parts, first the linear layer to compute 
z = w*h + b, then the sigmoid activaion for turning z into a probability. 
(p 177)
"""

import numpy as np 


class sigmoid:
    def __init__(self):
        pass 

    def forward(self, x):
        return 1 / (1 + np.exp(-x))