import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import os
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Define external variables
years = np.linspace(1050,1800,16,True, dtype = int)
realareayears = np.array([1200, 1300, 1400, 1450, 1500, 1600, 1650, 1700, 1750, 1800])
realpopyears = np.array([1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1800])
realareas = np.array([3270, 2780,2360,2360,1890,1400,910,530,160,0])
atot = 3563
areaprop = realareas/atot
totalPops = np.array([35, 43,52,48,55,62,69,76,84,93,106,192])*1000
cornishSpeakers = np.array([30,34,38,32,34,33,33,30,22,14,5, 0])*1000
popprop = (cornishSpeakers/totalPops)

# Load assets
sigma_smooth = 10
cornwallMask = np.load("../assets/cornwall_county_mask.npy").astype(bool)
initialMask = np.load("../assets/cornwall_river_mask.npy").astype(bool)
countyPopulation = np.load(f"../assets/cornwall_smoothed_dist_ss10.npy")
countyPopulation[~cornwallMask] = np.nan
popTot = np.nansum(countyPopulation[cornwallMask])
areaTot = np.nansum(cornwallMask.astype(int))
print(popTot)

initialProp = np.nansum(countyPopulation[initialMask])/np.nansum(countyPopulation[cornwallMask])
print("initial:", initialProp)

# Define hyperparams
alphas = [1.3, 1.7, 2.3]
factor = 1
sigma = 25
if sigma == 25:
    dt = 0.0004
elif sigma == 50:
    dt = 0.0001
chisqlist = []
resid17 = []

# Minimise MSE
fig, (axs, ax2) = plt.subplots(2, 1, figsize = (2.4, 2.9), gridspec_kw={'height_ratios': [3, 1]})
for number, alpha in enumerate(alphas):
    name_template = f"cornwallPopGaussian{sigma_smooth}ICriverAlpha{alpha}Beta1.1SigmavarFactor{factor}Deltat{dt}Tmax500.0"
    num = 0
    path = "../data"
    timespath = os.path.join(path, f"times_Gaussian_{name_template}_{num}.npy")
    times = np.load(timespath)

    poppath = os.path.join(path, f"pop1_Gaussian_2_{name_template}_{num}.npy")
    pops = np.load(poppath)

    prop = pops/popTot
    propinitial = prop[0]
    prop = prop*popprop[0]/propinitial
    lastIndex = np.argwhere(prop < 0.002)[0]
    prop = prop[0:lastIndex[0]+1]
    years = np.linspace(1200, 1800, lastIndex[0] + 1)

    chisq = 0
    for i, year in enumerate(realpopyears):
        index = (np.abs(years - year)).argmin()
        closestYear = years[index]
        Ei = prop[index]
        Oi = popprop[i]
        chisqContrib = ((Oi - Ei) ** 2) / Ei
        chisq += chisqContrib
        if number == 1:
            resid17.append(Ei-Oi)
    chisqlist.append(chisq)

    a = axs.plot(years, prop)#, label = fr"$\alpha = {alpha}$")
    colour = a[-1].get_color()


size=11
axs.tick_params(labelsize=size)
axs.xaxis.offsetText.set_fontsize(size)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.set_ylim(top = 1.0)

# Define and solve 'ODE model'
def dxdt(x, t, a, c, s):
    return (1-x)*c*(x**a)*s - c*x*(1-s)*(1-x)**a

def getFunction(x, x0, a, c, s):
    sol = odeint(dxdt, x0, realpopyears, args=(a, c, s))
    return sol[:,0]

params0 = [2,0.0000000000001,0.4]
params = [0.81135037, 1.00080347, 7.58154597, 0.49949072]
f = getFunction(realpopyears, *params)
chisq = 0
ODEresids = []
for i, year in enumerate(realpopyears):
    Ei = f[i]
    Oi = popprop[i]
    chisqContrib = ((Oi - Ei) ** 2) / Ei
    ODEresids.append(Ei-Oi)
    chisq += chisqContrib
ode = axs.plot(realpopyears, f, color = 'dimgrey', linestyle = "--", linewidth = 2)#,label = "ODE")

size = 11

historical = axs.plot(realpopyears, popprop, marker=".", color="r", linestyle="")  # , label = "Historical")
axs.set_ylabel("Population proportion", fontsize=size, labelpad=11.8)


# ax2 = fig.add_subplot(2, 1, 2, sharex = axs)
axs.set_xticklabels("")
ax2.tick_params(labelsize=size)
ax2.axhline(0, color = "dimgrey", linestyle = "--")
ax2.set_xlabel(f"Year", fontsize = size)
ax2.plot(realpopyears, ODEresids, marker = ".", color = "k", linestyle = "")
ax2.plot(realpopyears, resid17, marker = ".", color = "#ff7f0e", linestyle = "")
ax2.set_ylabel("Residuals", fontsize = size)
matplotlib.rcParams['axes.unicode_minus'] = False
ax2.set_yticks([-0.05, 0.0, 0.05])
plt.sca(axs)

###############################################################
# Some hacked-together code to make the multi-coloured legend #
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
l = axs.legend(handles=[lc,line1, line2], labels=[fr"$\alpha = 1.3, 1.7, 2.3$","Historical", "ODE"], handler_map={lc: HandlerColorLineCollection(numpoints=3)}, loc="upper right", framealpha=0, markerfirst=False)

# Uncomment to save
# plt.savefig(f"cornwallpop2.pdf", bbox_inches = "tight")

plt.show()

