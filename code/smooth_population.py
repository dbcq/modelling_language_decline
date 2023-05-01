import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from scipy import ndimage
from numba import jit


@jit(nopython = True)
def localAverageCountry(field):
    # Input should have a border of zeros around the edge - these are ignored
    # interior = field[1:-1, 1:-1]
    avgIncluded = field.copy()
    x, y = avgIncluded.shape
    for i in range(x):
        for j in range(y):
            sum = 0.
            counter = 0

            if includedRegion[i][j]:
                # Check the 4 nearest neighbours
                for dx, dy in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                    if includedRegion[i + dx][j + dy]:
                        # if (0 <= i + dx <= x - 1) and (0 <= j + dy <= y - 1):
                        sum += field[i + dx][j + dy]
                        counter += 1
                    else:
                        pass

                if counter == 0:
                    pass
                else:
                    avgIncluded[i][j] = sum / counter  # Counter makes sure we only divide by no. cells used

    return avgIncluded

def repeat(func, n, x):
     for i in range(n):
         x = func(x)
     return x


fig = plt.figure(figsize = (12,3))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

@jit(nopython=True)
def gauss(y, x, squarex, squarey, popTot):
    dist = np.hypot(x - squarex, y - squarey)
    A = popTot/(2*np.pi*sigma_smooth**2)
    density = A*np.exp(-dist**2/(2*sigma_smooth**2))
    return density


# Get smoothed distribution
includedRegion = np.load("../data/wales_mask.npy").astype(bool)
popDist = np.load("../data/wales_pop_dist.npy")

sizex, sizey = popDist.shape
x = np.arange(0, sizey, 1)
y = np.arange(0, sizex, 1)

X, Y = np.meshgrid(x, y)

smoothedPopDist = np.zeros_like(popDist)
sigma_smooth = 5

# Distribute population
for (x, y), el in np.ndenumerate(popDist):
    if (x == 20 and y == 1):
        im = ax2.imshow(smoothedPopDist, origin = "lower")
        plt.colorbar(im)
    popTot = el     
    print("popTot: ", popTot)
    g = 0
    if el == 0:
        pass
    else:
        n = 0
        for i in range(5):
            g += gauss(X, Y, x, y, popTot)
            outsidepop = np.sum(g[~includedRegion])
            g[~includedRegion] = 0
            popTot = outsidepop
            n+=1
        print("pop final: ", np.sum(g))
        print(n)
        smoothedPopDist += g
    print(x, ", ", y, end = "\r")

np.save("wales_smoothed_dist_ss5.npy", smoothedPopDist)
ax1.imshow(includedRegion, origin = "lower")
im = ax2.imshow(popDist, origin = "lower")
plt.colorbar(im)
plt.show()
