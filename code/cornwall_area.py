import numpy as np
from matplotlib import pyplot as plt
import os
from matplotlib import rcParams
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Define external data
years = np.linspace(1050,1800,16,True, dtype = int)
realareayears = np.array([1200, 1300, 1400, 1450, 1500, 1600, 1650, 1700, 1750, 1800])
realpopyears = np.array([1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1800])
realareas = np.array([3270, 2780,2360,2360,1890,1400,910,530,160,0])
atot = 3563
areaprop = realareas/atot
totalPops = np.array([35, 43,52,48,55,62,69,76,84,93,106,192])*1000
cornishSpeakers = np.array([30, 34,38,32,34,33,33,30,22,14,5, 0])*1000
popprop = (cornishSpeakers/totalPops)


sigma_smooth = 10
cornwallMask = np.load("../assets/cornwall_county_mask.npy").astype(bool)
initialMask = np.load("../assets/cornwall_river_mask.npy").astype(bool)
countyPopulation = np.load(f"../assets/cornwall_smoothed_dist_ss10.npy")
countyPopulation[~cornwallMask] = np.nan
popTot = np.nansum(countyPopulation[cornwallMask])
areaTot = np.nansum(cornwallMask.astype(int))

initialProp = np.nansum(countyPopulation[initialMask])/np.nansum(countyPopulation[cornwallMask])
print("initial:", initialProp)

# Define hyperparams
alphas = [1.3, 1.7, 2.3]
factor = 1
sigma = 25
if sigma == 25:
    dt = 0.0004
elif sigma == 50:
    dt = 0.0001
chisqlist = []
resid17 = []

# Calculate MSE
fig, axs = plt.subplots(1, 1, figsize = (2.4, 2.9))
for number, alpha in enumerate(alphas):
    name_template = f"cornwallPopGaussian{sigma_smooth}ICriverAlpha{alpha}Beta1.1SigmavarFactor{factor}Deltat{dt}Tmax500.0"
    num = 0
    path = "../data"
    timespath = os.path.join(path, f"times_Gaussian_{name_template}_{num}.npy")
    times = np.load(timespath)
    

    areapath = os.path.join(path, f"area1_Gaussian_2_{name_template}_{num}.npy")
    areas = np.load(areapath)
    areas = np.squeeze(areas)
    prop = areas/areaTot
    lastIndex = np.argwhere(prop == np.min(prop))[0]
    prop = prop[np.argmax(areas):lastIndex[0]+1]
    years = np.linspace(1200, 1800, lastIndex[0]+1-np.argmax(areas))
    times = times[np.argmax(areas):lastIndex[0]+1]

    chisq = 0
    for i, year in enumerate(realareayears):
        index = (np.abs(years - year)).argmin()
        closestYear = years[index]
        Ei = prop[index]
        Oi = areaprop[i]
        chisqContrib = ((Oi - Ei) ** 2) / Ei
        chisq += chisqContrib

    a = axs.plot(years, prop)

size=11
axs.tick_params(labelsize=size)
axs.xaxis.offsetText.set_fontsize(size)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.set_ylim(top = 1.0)


axs.plot(realareayears, areaprop, marker=".", color="r", linestyle="")  # , label = "Historical data")
axs.set_ylabel("Area proportion", fontsize=size)
axs.set_xlabel("Year")

###############################################################
# Some hacked-together code to make the multi-coloured legend #
###############################################################

from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib import colors

class HandlerColorLineCollection(HandlerLineCollection):
    def create_artists(self, legend, artist ,xdescent, ydescent,
                        width, height, fontsize,trans):
        x = np.linspace(0,width,self.get_numpoints(legend)+1)
        y = np.zeros(self.get_numpoints(legend)+1)+height/2.-ydescent
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap,
                     transform=trans)
        lc.set_array(x)
        lc.set_linewidth(artist.get_linewidth())
        return [lc]

t = np.linspace(1200, 1300, 4)
x = np.cos(np.pi * t)
y = np.sin(t)
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
cmap = colors.ListedColormap([plt.cm.tab10(i) for i in range(3)])
lc = LineCollection(segments, cmap=cmap,
                    norm=plt.Normalize(0, 10), linewidth=3)
lc.set_array(realpopyears)

axs.add_collection(lc)

line1 = Line2D([0], [0], color = 'red', marker='.', linestyle='')
line2 = Line2D([0], [0], color = 'dimgrey', linestyle='--')

l = axs.legend(handles=[lc, line1], labels =[fr"$\alpha = 1.3, 1.7, 2.3$", "Historical"], handler_map = {lc: HandlerColorLineCollection(numpoints=3)}, framealpha=0, markerfirst=False)

# Uncomment to save
# plt.savefig(f"cornwallarea2.pdf", bbox_inches = "tight")

plt.show()

