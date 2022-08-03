import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import cities
import os
import pdb


def plotheatmap(m_k, k):
    # Clear current plot figure
    ax2.clear()
    # plt.clf()
    plt.title(f"Memory at t = {k * delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    ax1.contourf(m_k, cmap=plt.cm.jet, vmin=0, vmax=1, alpha=0.5)

    if isogloss == "isogloss":
        isogloss_loc = np.where(np.abs(m_k - 0.5) < 0.06)
        isogloss_array = np.zeros_like(m_k)
        isogloss_array.fill(np.nan)
        isogloss_array[isogloss_loc] = 255
        # ax2.imshow(masked_m, interpolation = "none", cmap = "Greys", origin = "lower")
        ax2.imshow(isogloss_array, cmap="binary", origin="lower", vmin=0., vmax=255)

    print(k + 1, " out of ", len(ns), " plotted", end="\r")

    return plt


def animate(k):
    plotheatmap(m[k, :, :], k)


def initialCondition(memoryField, city):
    """Implements the initial condition that a particular city uses one variant"""
    [cityx, cityy] = city.pos
    r = city.radius
    x = np.arange(0, map_size, delta_x)
    y = np.arange(0, map_size, delta_x)
    X, Y = np.meshgrid(x, y)
    sqdist = (X - cityx) ** 2 + (Y - cityy) ** 2
    memoryField[:, :, :] = np.exp(-sqdist / ((radfactor * r) ** 2))

    # Create a solid mask
    ys, xs = np.ogrid[-cityy:map_size - cityy,
             -cityx:map_size - cityx]  # Create an openGrid containing the distances of each point to the city's centre
    region = xs ** 2 + ys ** 2 <= (initialfactor * 2) ** 2
    memoryField[:, region] = 1.0

    return memoryField


map_size = 400
max_iter_time = 350
alpha = 0.95
# alphas = [1.0, 1.005, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07]
beta = 1.1
sigma = 4
delta_x = 1
radfactor = 0.9
initialfactor = 3 / 4
delta_t = 0.0001  # (delta_x**2)/(4*alpha)
plot_type = "contour"

# pdb.set_trace()
#alphas = [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.2, 1.4, 1.6, 1.8, 2.2]
#alphas = [1.2, 1.3, 1.4]
#nums = [4, 0, 1] 

alphas = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]
nums = [1, 2, 10]
for number, alpha in enumerate(alphas):
    #name_template = f"cornwallPopGaussian10ICriverAlpha{alpha}Beta1.36Sigma50Deltat0.0001Tmax500.0"
    name_template = f"walesICbook5Alpha{alpha}Beta1.1SigmavarFactor2Deltat0.0004Tmax500.0"
    #name_template = f"cornwallNoPopTestICriverAlpha{alpha}Beta1.1Sigma50.0Deltat0.0001Tmax500.0"

    #num = nums[number]
    num = "1"
    interval = 500
    country = "wales"

    numFiles = len(os.listdir(f"/ddn/data/tffd79/walesGaussianPopDist/{name_template}_{num}_"))
    filenames = os.listdir(f"/ddn/data/tffd79/walesGaussianPopDist/{name_template}_{num}_")

    #numFiles = len(os.listdir(f"/ddn/data/tffd79/{name_template}_{num}"))
    #filenames = os.listdir(f"/ddn/data/tffd79/{name_template}_{num}")

    
    ns = np.array([filename[filename.rfind("_") + 1:-4] for filename in filenames]).astype(int)
    ns = np.sort(ns)
    print(ns)

    np.save(f"/ddn/data/tffd79/walesGaussianPopDist/popAreas/times_Gaussian_{name_template}_{num}.npy", ns)
    #print(ns)
    
    # q = np.load(f"data/Jan/{country}/{name_template}_{num}/{name_template}MEMORY_{num}_4000.npy")
    q = np.load(f"/ddn/data/tffd79/walesGaussianPopDist/{name_template}_{num}_/{filenames[0]}")

    #q = np.load(f"/ddn/data/tffd79/{name_template}_{num}/{filenames[0]}")
    mapsizex, mapsizey = q.shape
    m = np.zeros((len(ns), mapsizex, mapsizey), dtype=np.float32)
    
    for i, n in enumerate(ns):
        # m[i,:,:] = np.load(f"data/Jan/Cornwall/cornwallPopTestAlpha1.0Beta1.1Sigma50.0Deltat0.0001Tmax100.0_0/cornwallPopTestAlpha1.0Beta1.1Sigma50.0Deltat0.0001Tmax100.0MEMORY_0_{n}.npy")
        try:
            m[i, :, :] = np.load(f"/ddn/data/tffd79/walesGaussianPopDist/{name_template}_{num}_/{name_template}MEMORY_{nums[0]}_{n}.npy")
            #m[i, :, :] = np.load(f"/ddn/data/tffd79/{name_template}_{num}/{name_template}MEMORY_{nums[0]}_{n}.npy")

        except FileNotFoundError:
            try:
                m[i, :, :] = np.load(f"/ddn/data/tffd79/cornwallGaussianPopDist/{name_template}_{num}/{name_template}MEMORY_{nums[1]}_{n}.npy")
            except FileNotFoundError:
                m[i, :, :] = np.load(f"/ddn/data/tffd79/cornwallGaussianPopDist/{name_template}_{num}/{name_template}MEMORY_{nums[2]}_{n}.npy")

    print("Files loaded")

    
    
    walesMask = np.load("Wales_data/wales_fullsize_country_mask.npy").astype(bool)
    #countyPopulation = np.load("Cornwall_data/cornwall_smoothed_pop_dist2.npy")
    countryPopulation = np.load("Wales_data/smoothed_PopDistnew5_2.npy")
    countryPopulation[~walesMask] = np.nan
    flatPop = np.ones((countryPopulation.shape[0], countryPopulation.shape[1]))
    flatPop[~walesMask] = np.nan
    flatPopTot = np.nansum(flatPop[walesMask])
    popTot = np.nansum(countryPopulation[walesMask])
    areaTot = np.nansum(walesMask.astype(int))
    
    pop1 = []  # Population using variant 1
    area1 = []
    nopop_pop1 = []
    
    for i, n in enumerate(ns):
        locs = np.where(m[i, :, :] > 0.5)
        pop1.append(np.nansum(countryPopulation[locs]))
        area1.append(locs[0].shape)
        nopop_pop1.append(np.nansum(flatPop[locs]))
        
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(ns, pop1, marker=".", color="k", linestyle="")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Population")
    ax2.plot(ns, pop1 / popTot * 100, marker=".", color="k", linestyle="")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("% population")
    
    #plt.fig(f"/ddn/data/tffd79/cornwallGaussianPopDist/popAreas/nopoppop_area_{name_template}_{num}.png", bbox_inches="tight")
    
    np.save(f"/ddn/data/tffd79/walesGaussianPopDist/popAreas/pop1_Gaussian_2_{name_template}_{num}.npy", pop1)
    np.save(f"/ddn/data/tffd79/walesGaussianPopDist/popAreas/area1_Gaussian_2_{name_template}_{num}.npy", area1)
    #np.save(f"/ddn/data/tffd79/cornwallGaussianPopDist/popAreas/flat1_2_{name_template}_{num}.npy", nopop_pop1)


# isoglossRadius = []

# fig, ax1 = plt.subplots(1, 1)  #, sharex=True)
# ax2 = fig.add_subplot(111, facecolor="none")
# ax1.set_aspect('equal')
# ax2.set_aspect('equal')
# ax1.axis("off")
# ax2.axis("off")
#
# ax1.imshow(populationDensity, origin = "lower")
# ax2.contourf(m[0,:,:], alpha = 0.1)
# plt.show()

# isogloss = "isogloss"

# anim = animation.FuncAnimation(fig, animate, interval = 1, frames = len(ns), repeat = False)
# anim.save(f"graphs/Dec/paperexample11citiesbeta{beta}deltat{delta_t}sigma{sigma}t{max_iter_time}cityradius{round(city1.radius, 2)}k{radfactor}initial{initialfactor}contourEVERY10th.gif")
# anim.save(f"graphs/Jan/cornwallalpha0.95example_new2{isogloss}.gif")
