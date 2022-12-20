from numpy import *
from matplotlib import pyplot as plt

def mandelbrot(h, w, maxit = 40):
    y,x = ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
    c = x + y * 1j
    z = c
    divtime = maxit + zeros(z.shape, dtype=int)
    for i in range(maxit):
        z = z**2 + c
        diverge = z*conj(z) > 2**2
        div_now = diverge & (divtime == maxit)
        divtime[div_now] = i + 100
        z[diverge] = 2
    return divtime

fig = plt.subplots(1, figsize = (20,20))
s = "Mandelbrot Set"
plt.title(s)
plt.imshow(mandelbrot(1000,1000))
plt.axis('off')
plt.show()