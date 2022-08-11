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
                # print(i,j)
                # Check the 4 nearest neighbours
                for dx, dy in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                    if includedRegion[i + dx][j + dy]:
                        # if (0 <= i + dx <= x - 1) and (0 <= j + dy <= y - 1):
                        sum += field[i + dx][j + dy]
                        # print(field[i + dx][j + dy])
                        counter += 1
                    else:
                        pass
                # print(sum)

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
# ax3 = fig.add_subplot(1,3,3)
# ax4 = fig.add_subplot(1,2,2)

def getPopulationDensity(cities, sizex, sizey, delta_x):
    x = np.arange(0, sizey, delta_x)
    y = np.arange(0, sizex, delta_x)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((sizex, sizey))

    for city in cities:
        coords = city.pos
        Z += cityFunc(X, Y, *coords, city.radius)

    return Z


@jit(nopython=True)
def gauss(y, x, squarex, squarey, popTot):
    dist = np.hypot(x - squarex, y - squarey)
    A = popTot/(2*np.pi*sigma_smooth**2)
    density = A*np.exp(-dist**2/(2*sigma_smooth**2))
    # plt.imshow(density, origin = "lower")
    # plt.show()
    return density


# Get smoothed distribution
includedRegion = np.load("countries/Wales/wales_plus_mask.npy").astype(bool)
popDist = np.load("countries/Wales/wales_pop.npy")
# popDist = np.zeros((50, 100))
# popDist[0:,:] = range(0, 50)
# popDist[10, 20] = 50
sizex, sizey = popDist.shape
x = np.arange(0, sizey, 1)
y = np.arange(0, sizex, 1)

X, Y = np.meshgrid(x, y)



smoothedPopDist = np.zeros_like(popDist)
sigma_smooth = 10


for (x, y), el in np.ndenumerate(popDist):
    if (x == 20 and y == 1):
        im = ax2.imshow(smoothedPopDist, origin = "lower")
        plt.colorbar(im)
        plt.savefig("data/Feb/Cornwall/walestest.png", bbox_inches = "tight")
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



# np.save("countries/Cornwall_data/smoothed_gaussian.npy", smoothedPopDist)
ax1.imshow(includedRegion, origin = "lower")
im = ax2.imshow(popDist, origin = "lower")
plt.colorbar(im)
plt.show()
