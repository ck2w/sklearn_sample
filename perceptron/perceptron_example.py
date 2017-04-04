__author__ = 'ken.chen'

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import perceptron

# data
x = np.array([[2, 1, 2, 5, 7, 2, 3, 6, 1, 2, 5, 4, 6, 5], [2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7]])
y = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1])

colormap = np.array(['r', 'k'])
plt.scatter(x[0], x[1], c=colormap[y], s=40)

# fit
net = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)
net.fit(x.T, y)

# plot
ymin, ymax = plt.ylim()
w = net.coef_[0]
k = -w[0] / w[1]
xx = np.linspace(ymin, ymax)
yy = k * xx - (net.intercept_[0]) / w[1]

plt.plot(xx, yy, 'k-')
plt.ylim([0, 8])

# prediction
xd = np.array([[1, 1, 10], [1, 6, 8]])
prediction = net.predict(xd.T)
plt.scatter(xd[0], xd[1], c=colormap[prediction], s=40)
plt.ylim([0, 10])

print 'end'