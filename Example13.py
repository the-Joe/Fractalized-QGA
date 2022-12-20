from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

# Main body of program
points = 10000  # The number of points to use.

X = []  # A list of x-coordinates
Y = []  # A list of y-coordinates

# Set the starting point
point = [0.5, 0.0]
X.append(point[0])
Y.append(point[1])


def new_point(p):
    r = np.random.uniform(0, 1)
    if r < 0.02:
        p = [0.5, 0.27 * p[1]]
    elif 0.02 <= r <= 0.17:
        p = [-0.139 * p[0] + 0.263 * p[1] + 0.57, 0.246 * p[0] + 0.224 * p[1] - 0.036]
    elif 0.17 < r <= 0.3:
        p = [0.17 * p[0] - 0.215 * p[1] + 0.408, 0.222 * p[0] + 0.176 * p[1] + 0.0893]
    elif 0.3 < r < 1.0:
        p = [0.781 * p[0] + 0.034 * p[1] + 0.1075, -0.032 * p[0] + 0.739 * p[1] + 0.27]

    return p


# Generate a large number of points
for i in range(points):
    point = new_point(point)
    X.append(point[0])
    Y.append(point[1])

# Plot the results
plt.scatter(X, Y, c='g', s=.05)
plt.axis('Off')
plt.axes().set_aspect('equal')
plt.title("Barnsley Fern")
plt.savefig("barnsley_fern.png")
plt.show()
