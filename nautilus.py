"""
Attempt at training a model on spiral pattern. 
Inspiration from: http://cs231n.github.io/neural-networks-case-study/.
"""

import numpy as np
import matplotlib.pyplot as plt
import random 


plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(0)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D))
y = np.zeros(N*K, dtype='uint8')
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim([-1,1])
plt.ylim([-1,1])
#plt.show()



"""Create a linear model for first attempt at fitting."""
from pynet.core.model import Model
from pynet.layers.fullyconnected import linear
from pynet.layers.log_softmax import LogSoftmax
from pynet.losses.nllloss import NLLLoss
from pynet.optimizers.optimizer import  sgd

"""
linear_model = Model(
    linear(2, 3, bias=True),
    LogSoftmax(input_size=3, axis=1)
)

loss_fn = NLLLoss()
optimizer = sgd(linear_model, lr=1e-0, weight_decay=1e-4)

for i in range(100):

    out = linear_model(np.array(X))
    if not isinstance(out, tuple):
        out = (out,)

    out += (np.expand_dims(np.array(y), axis=1),)
    loss = loss_fn(*out)
    optimizer.step(loss_fn.backwards())

    print(f"Iteration {i}, Loss: {loss}")

h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
xs = xx.ravel()
ys = yy.ravel()
Z = np.zeros((xs.shape[0]))
for idx, i in enumerate(range(xs.shape[0])):
    temp = linear_model(np.array([[xs[idx], ys[idx]]]))
    Z[idx] = np.argmax(temp, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

"""
linear_model = Model(
    linear(2, 100, bias=True),
    linear(100, 3, bias=True, activation=None),
    LogSoftmax(input_size=3, axis=1)
)

loss_fn = NLLLoss()
optimizer = sgd(linear_model, lr=1e-4, weight_decay=1e-3)

for i in range(1):
    out = linear_model(np.array(X))
    if not isinstance(out, tuple):
        out = (out,)

    out += (np.expand_dims(np.array(y), axis=1),)
    loss = loss_fn(*out)
    optimizer.step(loss_fn.backwards())
    print(f"Iteration {i}, Loss: {loss}")


# evaluate training set accuracy
scores = linear_model(X)
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
"""
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
xs = xx.ravel()
ys = yy.ravel()
Z = np.zeros((xs.shape[0]))
for idx, i in enumerate(range(xs.shape[0])):
    temp = linear_model(np.array([[xs[idx], ys[idx]]]))
    Z[idx] = np.argmax(temp, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#plt.show()
"""
