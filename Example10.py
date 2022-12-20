import numpy as np
import matplotlib.pyplot as plt

line = [0,1]
depth = 10

def cantor_set(line, level=0):
    plt.plot(line,[level,level], color="k", lw=5, solid_capstyle="butt")
    if level < depth:
        s = np.linspace(line[0], line[1],4)
        cantor_set(s[:2], level+1)
        cantor_set(s[2:], level+1)

cantor_set(line)
plt.gca().invert_yaxis()
plt.show()