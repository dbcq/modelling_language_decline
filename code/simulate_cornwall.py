"""Given a region, a population distribution and an initial condition, simulates
the decline of the language using the equation described in the article."""

import numpy as np
from PIL import Image
import os
from numba import jit
import sys


@jit(nopython=True)
def sigmoid(m, alpha = 1., beta = 1.):
    frequency = (m**(alpha*beta))/(m**(alpha*beta) + (1- m**alpha)**beta)
    return frequency

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


def calculate(m):
    """Solve the differential equation"""

    avgDensity = localAverageCountry(populationDensity)

    for k in range(0, iterations-1, 1):
        if k % saveInterval == 0:
            filename = f"cornwallPopGaussian{sigma_smooth}ICriverAlpha{alpha}Beta{beta}SigmavarFactor2Deltat{delta_t}Tmax{tmax}MEMORY_{folder_num}_{k+offset}.npy"
            np.save(os.path.join(folder_name, filename), m)

        print(k+1, " out of ", iterations, end = "\r")
        avgDensFreq = localAverageCountry(populationDensity*f[:,:])
        dmdt = np.zeros((mapsizex, mapsizey))
        dmdt[includedRegion] = (f[includedRegion] - m[includedRegion]) + (2*sigma[includedRegion]**2) * (avgDensFreq[includedRegion]/avgDensity[includedRegion] - f[includedRegion])

        m[includedRegion] = m[includedRegion] + dmdt[includedRegion]*delta_t
        f[includedRegion] = sigmoid(m[includedRegion], alpha, beta)
    return m


if __name__ == "__main__":

    populationDensity = np.load(f"cornwall_smoothed_dist_ss10.npy")

    alpha = 1.1
    sigma_smooth = 10
    factor = np.log(100)/10
    beta = 1.1
    sigma_coeff = 25
    sigma = sigma_coeff*(1-np.exp(-factor*populationDensity))
    tmax = 500.0
    delta_t = 0.0004
    delta_x = 1.0
    offset = 0
    initialfactor = 0.5
    iterations = int(tmax/delta_t)
    saveInterval = 2000

    # Load pop dist and included region
    includedRegionImg = Image.open("cornwall_mask.tif")
    includedRegionArray = np.array(includedRegionImg, dtype = bool)
    includedRegion = np.zeros((includedRegionArray.shape[0] + 2, includedRegionArray.shape[1] + 2), dtype = bool)
    includedRegion[1:-1,1:-1] = includedRegionArray
    includedRegion = np.flip(includedRegion, axis = 0)
    mapsizex, mapsizey = includedRegion.shape

    print("mask loaded")


    # Initial condition
    initialRegion = np.load("cornwall_river_mask.npy").astype(bool)
    m = np.zeros((mapsizex, mapsizey))
    m[:,:int(mapsizex/2)] = 1.0
    m[~includedRegion] = np.nan
    m[initialRegion] = 1.0

    f = np.zeros((mapsizex, mapsizey))
    f = sigmoid(m, alpha, beta)

    print("m and f initialised")

    folder_num = 0
    while True:
        try:
            folder_name = f"cornwallPopGaussian{sigma_smooth}ICriverAlpha{alpha}Beta{beta}SigmavarFactor2Deltat{delta_t}Tmax{tmax}_{folder_num}"
            os.mkdir(folder_name)
            break
        except OSError:
            folder_num+=1

    print("folder created")
    
    m = calculate(m)