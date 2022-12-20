import numpy as np
import matplotlib.pylab as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x, step):
    return (sigmoid(x+step) - sigmoid(x)) / step

x = np.linspace(-10, 10, 1000)

y1 = sigmoid(x)
y2 = derivative(x, 0.0000000000001)

plt.plot(x, y1, label='sigmoid')
plt.plot(x, y2, label='derivative')
plt.legend(loc='upper left')
plt.show()