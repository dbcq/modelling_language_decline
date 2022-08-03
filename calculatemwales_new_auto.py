import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from PIL import Image, ImageDraw
import os
from numba import jit
import sys

class City:
    def __init__(self, pos, size, mem):
        self.pos = np.array(pos)  # Should be a numpy array in form [x, y]
        self.radius = size
        self.mem = mem  # Frequency of usage of variant A


def getPopulationDensity(cities, sizex, sizey, delta_x):
    x = np.arange(0, sizey, delta_x)
    y = np.arange(0, sizex, delta_x)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((sizex, sizey))

    for city in cities:
        coords = city.pos
        Z += cityFunc(X, Y, *coords, city.radius)

    return Z


def cityFunc(x, y, cityx, cityy, R):
    dist = np.hypot(x - cityx, y - cityy)
    density = np.exp(-dist**2/R**2)
    return density


def sigmoid(m, alpha = 1., beta = 1.):
    frequency = (m**(alpha*beta))/(m**(alpha*beta) + (1- m**alpha)**beta)
    return frequency



def localAverage(field):
    # Input should have a border of zeros around the edge - these are ignored
    interior = field[1:-1, 1:-1]
    avgInterior = interior.copy()
    x, y = interior.shape
    for i in range(x):
        for j in range(y):
            sum = 0.
            counter = 0

            # Check the 4 nearest neighbours
            for dx, dy in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                if (0 <= i + dx <= x - 1) and (0 <= j + dy <= y - 1):
                    sum += interior[i + dx][j + dy]
                    # print(field[i + dx][j + dy])
                    counter += 1
                else:
                    pass
            # print(sum)
            avgInterior[i][j] = sum / counter  # Counter makes sure we only divide by no. cells used

    result = np.zeros((x + 2, y + 2))
    result[1:-1, 1:-1] = avgInterior
    return result

@jit(nopython=True)
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


def calculate(m):
    avgDensity = localAverageCountry(populationDensity)

    for k in range(0, iterations-1, 1):
        if k % saveInterval == 0:
            #filename = f"walesICbook{sigma_smooth}Alpha{alpha}Beta{beta}SigmavarFactor{args[4]}Deltat{delta_t}Tmax{tmax}MEMORY_{folder_num}_{k}.npy"
            #filename = f"walesICbook{sigma_smooth}Alpha{alpha}Beta{beta}Sigma{sigma}Deltat{delta_t}Tmax{tmax}MEMORY_{folder_num}_{k}.npy"
            filename = f"walesICbookNoPopAlpha{alpha}Beta{beta}Sigma{sigma}Deltat{delta_t}Tmax{tmax}MEMORY_{folder_num}_{k}.npy"

            np.save(os.path.join(folder_name, filename), m)

        print(k+1, " out of ", iterations, end = "\r")
        avgDensFreq = localAverageCountry(populationDensity*f[:,:])
        dmdt = np.zeros((mapsizex, mapsizey))
        #dmdt[includedRegion] = (f[includedRegion] - m[includedRegion]) + (2*sigma[includedRegion]**2) * (avgDensFreq[includedRegion]/avgDensity[includedRegion] - f[includedRegion])
        dmdt[includedRegion] = (f[includedRegion] - m[includedRegion]) + (2*sigma**2) * (avgDensFreq[includedRegion]/avgDensity[includedRegion] - f[includedRegion])


        m[includedRegion] = m[includedRegion] + dmdt[includedRegion]*delta_t
        f[includedRegion] = sigmoid(m[includedRegion], alpha, beta)
    return m

def initialCondition(memoryField, city):
    """Implements the initial condition that a particular city uses one variant"""
    [cityx, cityy] = city.pos
    r = city.radius
    x = np.arange(0, mapsizex, delta_x)
    y = np.arange(0, mapsizey, delta_x)
    X, Y = np.meshgrid(x, y)
    # sqdist = (X-cityx)**2 + (Y-cityy)**2
    # memoryField[:, :, :] = np.exp(-sqdist/(radfactor*(r**2)))

    # Create a solid mask
    ys, xs = np.ogrid[-cityy:mapsizey-cityy, -cityx:mapsizex-cityx]  # Create an openGrid containing the distances of each point to the city's centre
    region = xs**2 + ys**2 <= (initialfactor*r)**2
    memoryField[:, region] = 1.0

    return memoryField


args = sys.argv
country = args[1]
alpha = float(args[2])
sigma_smooth = int(args[3])
#factor = int(args[4])
#if factor == 1:
#    factor = np.log(2)/10
#elif factor == 2:
#    factor = np.log(100)/10


if country == "wales":
    
    #includedRegionImg = Image.open("Cornwall_data/cornwall_mask.tif")
    includedRegion = np.load("Wales_data/wales_plus_mask.npy").astype(bool)
    
    #includedRegionArray = np.array(includedRegionImg, dtype = bool)
    #includedRegion = np.zeros((includedRegionArray.shape[0] + 2, includedRegionArray.shape[1] + 2), dtype = bool)
    #includedRegion[1:-1,1:-1] = includedRegionArray
    #includedRegion = np.flip(includedRegion, axis = 0)
    
    mapsizex, mapsizey = includedRegion.shape
    
    #populationDensity = np.load(f"Wales_data/smoothed_PopDistnew{sigma_smooth}_2.npy")
    
    populationDensity = np.ones((mapsizex, mapsizey))
    #print(includedRegion.shape)
    #populationDensity[~includedRegion] = 0
    
    print("mask loaded")
    
    beta = 1.1
    sigma_coeff = 25 #50.0

    #sigma = sigma_coeff*(1-np.exp(-factor*populationDensity))#
    sigma = 25
    # mapsizex = 100
    # mapsizey = 200
    tmax = 500.0
    delta_t = 0.0004
    delta_x = 1.0
    
    initialfactor = 0.5
    
    iterations = int(tmax/delta_t)
    
    # populationDensity_shown = populationDensity.copy()
    # populationDensity_shown[excludedRegion] = np.nan
    # np.save("data/Dec/popDistUniform400x400.npy", populationDensity)
    
    m = np.zeros((mapsizex, mapsizey))
    # m[0,:,:] = np.random.uniform(0, 1, (mapsiyze, mapsize))
    # m = initialCondition(m, city1)
    initialRegion = np.load("/ddn/home/tffd79/Wales_data/wales_initial_1850.npy").astype(bool)
    m[180:,:350] = includedRegion[180:, :350]
    m[initialRegion] = 1.0
    m[~includedRegion] = np.nan
    
    
    f = np.zeros((mapsizex, mapsizey))
    f = sigmoid(m, alpha, beta)
    
    print("m and f initialised")
    
    folder_num = 0
    while True:
        print(folder_num)
        try:
            #folder_name = f"/ddn/data/tffd79/walesGaussianPopDist/walesICbook{sigma_smooth}Alpha{alpha}Beta{beta}SigmavarFactor{args[4]}Deltat{delta_t}Tmax{tmax}_{folder_num}"
            folder_name = f"/ddn/data/tffd79/walesGaussianPopDist/walesICbook{sigma_smooth}Alpha{alpha}Beta{beta}Sigma{sigma}Deltat{delta_t}Tmax{tmax}_{folder_num}"
            folder_name = f"/ddn/data/tffd79/walesGaussianPopDist/walesICbookNoPopAlpha{alpha}Beta{beta}Sigma{sigma}Deltat{delta_t}Tmax{tmax}_{folder_num}"

            
            #folder_name = f"data/Jan/Cornwall/cornwallTestAlpha{alpha}Beta{beta}Sigma{sigma}Deltat{delta_t}Tmax{tmax}_{folder_num}"
            os.mkdir(folder_name)
            break
        except OSError:
            folder_num+=1
    
    print("folder created")
    
    saveInterval = 500
    
    m = calculate(m)
    # m[~includedRegion] = np.nan
    
    
    # m = m[::10,:,:] # Reduce size by factor of 10
    # print("hello")
    # plot = False
    # np.save(f"data/Jan/Country/cornwallTestAlpha{alpha}Beta{beta}Sigma{sigma}Deltat{delta_t}Tmax{tmax}MEMORY_every10th.npy", m)
    #np.save(f"/ddn/data/tffd79/startingagainMapsize400Alpha{alpha}Beta{beta}Sigma{sigma}Deltat{delta_t}Tmax{tmax}MEMORY.npy", m)
    
    if plot == True:
        def next(k):
            plotheatmap(m[k, :, :], k)
    
    
        def plotheatmap(m, k):
            ax2.clear()
            ax2.imshow(m, origin="lower", alpha = 0.5)
            plt.title(f"t = {10*k}")
    
    
        fig, ax1 = plt.subplots(1, 1)  # , sharex=True)
        ax2 = fig.add_subplot(111, facecolor="none")
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax1.axis("off")
        ax2.axis("off")
        ax1.imshow(populationDensity, origin = "lower")
    
        # m = np.load(f"data/Jan/UKtest/UKtest4Alpha1.0Beta1.2Sigma4.0Deltat0.01Tmax0.1MEMORY_0.npy")
        # print(m.shape)
        ani = animation.FuncAnimation(fig, next, frames = m.shape[0])
        # ani.save("data/Jan/solvingPDEmetropolitanSciPywTEST3.gif")





