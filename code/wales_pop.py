import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import os
import matplotlib
from mpl_toolkits.axes_grid.inset_locator import InsetPosition
                                                  
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})                                                  


years = np.linspace(1050,1800,16,True, dtype = int)

realpopyears = [1901, 1911, 1921, 1931, 1951, 1961, 1971, 2001]
realPop = [49.9, 43.5, 37.1, 36.8, 28.9, 26.0, 20.9, 20.5]
popprop = np.array(realPop)/100

walesMask = np.load("wales_fullsize_country_mask.npy").astype(bool)
initialMask = np.load("wales_initial_1850.npy").astype(bool)
walesPopulation = np.load(f"wales_smoothed_dist_ss10.npy")
walesPopulation[~walesMask] = np.nan
popTot = np.nansum(walesPopulation[walesMask])
areaTot = np.nansum(walesMask.astype(int))
print(popTot)

initialProp = np.nansum(walesPopulation[initialMask])/np.nansum(walesPopulation[walesMask])
print("initial:", initialProp)

# alphas = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]
alphas = [1.2, 1.7, 2.3]
sigma = 25
endyear = 4087

fig, axs = plt.subplots(1, 1, figsize = (3.3, 3.1))#, gridspec_kw={'height_ratios': [3, 1]})
ax2 = plt.axes([0,0,1,1])
for number, alpha in enumerate(alphas):
    name_template = f"walesICbook5Alpha{alpha}Beta1.1SigmavarFactor1Deltat0.0004Tmax500.0"
    num = 1
    path = "data"
    timespath = os.path.join(path, f"times_Gaussian_{name_template}_{num}.npy")
    times = np.load(timespath)

    poppath = os.path.join(path, f"pop1_Gaussian_2_{name_template}_{num}.npy")
    pops = np.load(poppath)

    prop = pops/popTot
    propinitial = prop[0]
    lastIndex = np.argwhere(prop < 0.002)[0]
    if alpha == 1.7:
        print(lastIndex)
    prop = prop[0:lastIndex[0]+1]
    years = np.linspace(1850, endyear, lastIndex[0] + 1)


    chisq = 0
    for i, year in enumerate(realpopyears):
        index = (np.abs(years - year)).argmin()
        closestYear = years[index]
        Ei = prop[index]
        print("Oi :", Ei)
        Oi = popprop[i]
        print("Ei:", Oi)
        chisqContrib = ((Oi - Ei) ** 2) / Ei
        chisq += chisqContrib

    MSE = 0
    N = popprop.shape[0]
    for i, year in enumerate(realpopyears):
        index = (np.abs(years - year)).argmin()
        Ei = prop[index]
        Oi = popprop[i]
        MSEcontrib = (Oi - Ei) ** 2
        MSE += MSEcontrib
    MSE = MSE / N

    a = axs.plot(years, prop)#, label = MSE)
    a = ax2.plot(years, prop)#, label = MSE)


def dxdt(x, t, a, c, s):
    print(a, c, s, end="\r")
    return (1-x)*c*(x**a)*s - c*x*(1-s)*(1-x)**a

def getFunction(x, x0, a, c, s):
    sol = odeint(dxdt, x0, realpopyears, args=(a, c, s))
    return sol[:,0]

params = [0.4963, 1.0073, -3.5302, 0.5035]
f = getFunction(realpopyears, *params)
chisq = 0
ODEresids = []
for i, year in enumerate(realpopyears):
    Ei = f[i]
    print("Oi :", Ei)
    Oi = popprop[i]
    print("Ei:", Oi)
    chisqContrib = ((Oi - Ei) ** 2) / Ei
    chisq += chisqContrib

MSE = 0

for i, year in enumerate(realpopyears):
    Ei = f[i]
    print("Oi :", Ei)
    Oi = popprop[i]
    print("Ei:", Oi)
    MSEcontrib = (Oi - Ei) ** 2

    MSE += MSEcontrib
MSE = MSE/N
ax2.plot(realpopyears, f, color="dimgrey", linestyle="--",linewidth = 2)#,label = fr"$\chi^2 = {MSE}")
axs.plot(realpopyears, f, color="dimgrey", linestyle="--", linewidth = 2)#,label = fr"$\chi^2 = {MSE}")
axs.plot(realpopyears, popprop, marker = ".", color = "r",linestyle = "")#, label = "Historical data")
ax2.plot(realpopyears, popprop, marker = ".", color = "r",linestyle = "")#, label = "Historical data")


#############################################################
# Some hacked-together code to make the multi-coloured legend
#############################################################

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
lc.set_array(np.array(realpopyears))

ax2.add_collection(lc)

line1 = Line2D([0], [0], color = 'red', marker='.', linestyle='')
line2 = Line2D([0], [0], color = 'dimgrey', linestyle='--')

l = ax2.legend(handles=[lc, line1], labels =[fr"$\alpha = 1.3, 1.7, 2.3$", "Historical"], handler_map = {lc: HandlerColorLineCollection(numpoints=3)}, framealpha=0, markerfirst=False)
l = ax2.legend(handles=[lc,line1, line2], labels=[fr"$\alpha = 1.3, 1.7, 2.3$","Historical", "ODE"], handler_map={lc: HandlerColorLineCollection(numpoints=3)}, loc="upper right", framealpha=0, markerfirst=False)


# Create inset

ip = InsetPosition(axs, [0.3,0.4,0.7,0.6])
ax2.set_axes_locator(ip)
xmin = 1870
xmax = 2100
ymin = 0.13
ymax = 0.7# + (xmax-xmin)
extent = (xmin, xmax, ymin, ymax)
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)

size = 11
axs.set_xlabel("Year", fontsize = size)
axs.set_ylabel("Population proportion", fontsize = size)
axs.tick_params(labelsize=size)
axs.xaxis.offsetText.set_fontsize(size)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)

plt.savefig("wales_pop.pdf", bbox_inches = "tight")
plt.show()
