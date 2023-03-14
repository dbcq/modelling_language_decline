import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from PIL import Image, ImageDraw
import os
from numba import jit
import sys


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
                # print(i,j)
                # Check the 4 nearest neighbours
                for dx, dy in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                    if includedRegion[i + dx][j + dy]:
                        # if (0 <= i + dx <= x - 1) and (0 <= j + dy <= y - 1):
                        sum += field[i + dx][j + dy]
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
            filename = f"walesICbook{sigma_smooth}Alpha{alpha}Beta{beta}SigmavarFactor2Deltat{delta_t}Tmax{tmax}MEMORY_{folder_num}_{k}.npy"

            np.save(os.path.join(folder_name, filename), m)

        print(k+1, " out of ", iterations, end = "\r")
        avgDensFreq = localAverageCountry(populationDensity*f[:,:])
        dmdt = np.zeros((mapsizex, mapsizey))
        dmdt[includedRegion] = (f[includedRegion] - m[includedRegion]) + (2*    sigma[includedRegion]**2) * (avgDensFreq[includedRegion]/avgDensity[includedRegion] - f[includedRegion])

        m[includedRegion] = m[includedRegion] + dmdt[includedRegion]*delta_t
        f[includedRegion] = sigmoid(m[includedRegion], alpha, beta)
    
    return m

if __name__ == "__main__":

    includedRegion = np.load("code/wales_mask.npy").astype(bool)
    
    mapsizex, mapsizey = includedRegion.shape
    
    populationDensity = np.load(f"code/smoothed_PopDistnew5_2.npy")
    populationDensity[~includedRegion] = 0
    
    print("mask loaded")
    
    # Simulation params
    alpha = 2.5
    beta = 1.1
    sigma_coeff = 25
    sigma_smooth = 5
    factor = np.log(2)/10
    sigma = sigma_coeff*(1-np.exp(-factor*populationDensity))#
    tmax = 500.0
    delta_t = 0.0004
    delta_x = 1.0
    initialfactor = 0.5


    # File params
    plot = True
    saveInterval = 500
    reduce_size = 1    # Reduce size of m produced by taking only every (n>1)th frame
    
    iterations = int(tmax/delta_t)
    
    # Initialise memory and frequency fields
    m = np.zeros((mapsizex, mapsizey))
    initialRegion = np.load("code/wales_initial_1850.npy").astype(bool)
    m[180:,:350] = includedRegion[180:, :350]
    m[initialRegion] = 1.0
    m[~includedRegion] = np.nan
    
    f = np.zeros((mapsizex, mapsizey))
    f = sigmoid(m, alpha, beta)
    
    print("m and f initialised")
    
    # Create folder for data (iterating folder number if already exists)
    folder_num = 0
    while True:
        print(folder_num)
        try:
            folder_name = f"walesICbook{sigma_smooth}Alpha{alpha}Beta{beta}SigmavarFactor2Deltat{delta_t}Tmax{tmax}_{folder_num}"           
            os.mkdir(folder_name)
            break
        except OSError:
            folder_num+=1
    
    print("folder created")
    
    m = calculate(m)
    





