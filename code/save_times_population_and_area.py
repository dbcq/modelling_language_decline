import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

alphas = [1.2, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]
run_num = 0
for number, alpha in enumerate(alphas):

    # Uncomment according to country
    #name_template = f"cornwallPopGaussian10ICriverAlpha{alpha}Beta1.36Sigma50Deltat0.0001Tmax500.0"
    name_template = f"walesICbook5Alpha{alpha}Beta1.1SigmavarFactor2Deltat0.0004Tmax500.0"
    #name_template = f"cornwallNoPopTestICriverAlpha{alpha}Beta1.1Sigma50.0Deltat0.0001Tmax500.0"

    interval = 500
    country = "wales"  # "cornwall"

    numFiles = len(os.listdir(f"simulation_data/{name_template}_{run_num}"))
    filenames = os.listdir(f"simulation_data/{name_template}_{run_num}")
    
    ns = np.array([filename[filename.rfind("_") + 1:-4] for filename in filenames]).astype(int)
    ns = np.sort(ns)
    print(ns)

    # Save timestamps
    # np.save(f"data/times_Gaussian_{name_template}_{num}.npy", ns)
    
    initial = np.load(f"simulation_data/{name_template}_{run_num}/{filenames[0]}")

    mapsizex, mapsizey = initial.shape
    m = np.zeros((len(ns), mapsizex, mapsizey), dtype=np.float32)
    
    for i, n in enumerate(ns):
        m[i, :, :] = np.load(f"simulation_data/{name_template}_{run_num}/{name_template}MEMORY_{run_num}_{n}.npy")

    print("Files loaded")

    
    walesMask = np.load("assets/wales_country_mask.npy").astype(bool)
    #countyPopulation = np.load("assets/cornwall_smoothed_dist_ss10.npy")
    countryPopulation = np.load("assets/wales_smoothed_dist_ss5.npy")

    # Discount areas not included in the shape of the country (hence np.nansum later)
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
    
    np.save(f"data/pop1_Gaussian_2_{name_template}_{run_num}.npy", pop1)
    np.save(f"data/area1_Gaussian_2_{name_template}_{run_num}.npy", area1)
