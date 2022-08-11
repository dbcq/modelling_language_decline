import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from scipy.integrate import odeint
import os
from matplotlib import rcParams
# rcParams.update({'font.family': 'serif','text.usetex':True})
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
# matplotlib.use("pgf")


years = np.linspace(1050,1800,16,True, dtype = int)
# realyears = np.array([1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1800])
realareayears = np.array([1200, 1300, 1400, 1450, 1500, 1600, 1650, 1700, 1750, 1800])
realpopyears = np.array([1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1800])
# realareas = np.array([np.nan,3270,np.nan,2780,np.nan,2360,2360,1890,np.nan,1400,910,530,160,0])
realareas = np.array([3270, 2780,2360,2360,1890,1400,910,530,160,0])
atot = 3563
# atot = 3270
areaprop = realareas/atot
# totalPops = np.array([21, 28,35,43,52,48,55,62,69,76,84,93,106,192])*1000
totalPops = np.array([35, 43,52,48,55,62,69,76,84,93,106,192])*1000
# cornishSpeakers = np.array([20,26,30,34,38,32,34,33,33,30,22,14,5,0])*1000
cornishSpeakers = np.array([30, 34,38,32,34,33,33,30,22,14,5, 0])*1000
popprop = (cornishSpeakers/totalPops)


sigma_smooth = 10
cornwallMask = np.load("countries/Cornwall_data/cornwall_county_mask_w_border_rightway.npy").astype(bool)
initialMask = np.load("countries/Cornwall_data/cornwall_river_mask.npy").astype(bool)
countyPopulation = np.load(f"countries/Cornwall_data/smoothed_PopDistnew{sigma_smooth}_2.npy")
countyPopulation[~cornwallMask] = np.nan
popTot = np.nansum(countyPopulation[cornwallMask])
areaTot = np.nansum(cornwallMask.astype(int))
print(popTot)

initialProp = np.nansum(countyPopulation[initialMask])/np.nansum(countyPopulation[cornwallMask])
print("initial:", initialProp)

# alphas = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]
alphas = [1.3, 1.7, 2.3]
# alphas = [1.8]
ap = "a"
factor = 1
sigma = 25
if sigma == 25:
    dt = 0.0004
elif sigma == 50:
    dt = 0.0001
chisqlist = []
resid17 = []
# numbers = ["all", "all", "all", 0]
# fig, (axs, ax2) = plt.subplots(2, 1, figsize = (2.4, 2.9), gridspec_kw={'height_ratios': [3, 1]})
fig, axs = plt.subplots(1, 1, figsize = (2.4, 2.9))
for number, alpha in enumerate(alphas):
    name_template = f"cornwallPopGaussian{sigma_smooth}ICriverAlpha{alpha}Beta1.1SigmavarFactor{factor}Deltat{dt}Tmax500.0"
    # num = numbers[number]
    num = 0
    path = "data/Mar"
    timespath = os.path.join(path, f"times_Gaussian_{name_template}_{num}.npy")
    times = np.load(timespath)
    if ap == "p":
        poppath = os.path.join(path, f"pop1_Gaussian_2_{name_template}_{num}.npy")
        pops = np.load(poppath)

        prop = pops/popTot
        propinitial = prop[0]
        prop = prop*popprop[0]/propinitial
        lastIndex = np.argwhere(prop < 0.002)[0]
        if alpha == 1.5:
            print(lastIndex, ' last index')
        prop = prop[0:lastIndex[0]+1]
        years = np.linspace(1200, 1800, lastIndex[0] + 1)

        chisq = 0
        for i, year in enumerate(realpopyears):
            index = (np.abs(years - year)).argmin()
            closestYear = years[index]
            Ei = prop[index]
            # print("Oi :", Ei)
            Oi = popprop[i]
            # print("Ei:", Oi)
            chisqContrib = ((Oi - Ei) ** 2) / Ei
            chisq += chisqContrib
            if number == 4: #1:  4
                resid17.append(Ei-Oi)
        chisqlist.append(chisq)

    elif ap == "a":
        areapath = os.path.join(path, f"area1_Gaussian_2_{name_template}_{num}.npy")
        areas = np.load(areapath)
        areas = np.squeeze(areas)
        prop = areas/areaTot
        # prop = areas/np.max(areas) * 100
        lastIndex = np.argwhere(prop == np.min(prop))[0]
        prop = prop[np.argmax(areas):lastIndex[0]+1]
        years = np.linspace(1200, 1800, lastIndex[0]+1-np.argmax(areas))
        times = times[np.argmax(areas):lastIndex[0]+1]

        chisq = 0
        for i, year in enumerate(realareayears):
            index = (np.abs(years - year)).argmin()
            closestYear = years[index]
            Ei = prop[index]
            print("Oi :", Ei)
            Oi = areaprop[i]
            print("Ei:", Oi)
            chisqContrib = ((Oi - Ei) ** 2) / Ei
            chisq += chisqContrib

    # chisq /= len(realyears)-1
    # if number == 0 or number == len(alphas)-1:
    a = axs.plot(years, prop)#, label = fr"$\alpha = {alpha}$")
    # print(number, ap)
    if np.logical_and(True,ap == "p"):
        # print("YES HERE")
        colour = a[-1].get_color()
        axs.plot(years, prop * propinitial/prop[0], color="grey", alpha=0.3, linestyle="--")
    # else:
    # a = axs.plot(years, prop)
    # colour = a[-1].get_color()
    # a = axs.plot(years, prop * popprop[0] / prop[0], color=colour, alpha=0.5, linestyle=(0, (5, 10)))




# straightLine = -91.78/600 * (realpopyears-1200) + 91.78
# chisqStraightLine = np.sum([(popprop[i]-straightLine[i])**2 / straightLine[i] for i in range(len(straightLine))])
#
# axs.plot(realpopyears, straightLine, color = "k", marker = "", linestyle = "-", alpha = 0.5, label = fr"$\chi^2 = {int(round(chisq))}$")


if ap == "a":
    ii = "(i)"
elif ap == "p":
    ii = "(ii)"
size=11
axs.tick_params(labelsize=size)
axs.xaxis.offsetText.set_fontsize(size)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.set_ylim(top = 1.0)


# plt.legend(fontsize = size)
# plt.legend()
# plt.ylim(0, 100)
# plt.savefig("Report/chisquaring.png", bbox_inches = "tight")

#
# fig2, ax = plt.subplots(1,1)
# ax.plot(alphas, chisqlist)
if ap == "p":
    def dxdt(x, t, a, c, s):
        # print(a, c, s, end="\r")
        return (1-x)*c*(x**a)*s - c*x*(1-s)*(1-x)**a

    def getFunction(x, x0, a, c, s):
        # sol = solve_ivp(dxdt, [wyears[0], wyears[-1]], [wprop[0]], t_eval = wyears, args=(a, c, s))
        sol = odeint(dxdt, x0, realpopyears, args=(a, c, s))
        # sol = np.array([])
        return sol[:,0]

    params0 = [2,0.0000000000001,0.4]

    # popt, pcov = curve_fit(getFunction, realpopyears, popprop, bounds = ([0., 0., -np.inf, 0.], [np.inf, np.inf, np.inf, 1.]))

    # print(popt)
    # sol = solve_ivp(dxdt, t_span = [years[0], years[-1]], y0=[0.9], args = (1.3, 0.0001, 0.3))

    # plt.clf()
    # plt.plot
    params = [0.81135037, 1.00080347, 7.58154597, 0.49949072]
    f = getFunction(realpopyears, *params)
    chisq = 0
    ODEresids = []
    for i, year in enumerate(realpopyears):
        Ei = f[i]
        # print("Oi :", Ei)
        Oi = popprop[i]
        # print("Ei:", Oi)
        chisqContrib = ((Oi - Ei) ** 2) / Ei
        ODEresids.append(Ei-Oi)
        chisq += chisqContrib
    ode = axs.plot(realpopyears, f, color = 'dimgrey', linestyle = "--", linewidth = 2)#,label = "ODE")

    size = 11
    # realareas/3563 *100
if ap == "p":
    historical = axs.plot(realpopyears, popprop, marker=".", color="r", linestyle="")  # , label = "Historical")
    axs.set_ylabel("Population proportion", fontsize=size, labelpad=11.8)
elif ap == "a":
    axs.plot(realareayears, areaprop, marker=".", color="r", linestyle="")  # , label = "Historical data")
    axs.set_ylabel("Area proportion", fontsize=size)
    axs.set_xlabel("Year")
    # print("chi^2 = ", chisq)

    # print(popt)
    # perr = np.sqrt(np.diag(pcov))
    # print(perr)
    # plt.legend()
    # plt.plot(sol)
    # plt.show()

if ap == "p":
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
    # axs.set_xlabel("Year", fontsize=size)

from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib import colors
#
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
#
#
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
#
line1 = Line2D([0], [0], color = 'red', marker='.', linestyle='')
line2 = Line2D([0], [0], color = 'dimgrey', linestyle='--')
#
#
l = axs.legend(handles=[lc, line1], labels =[fr"$\alpha = 1.3, 1.7, 2.3$", "Historical"], handler_map = {lc: HandlerColorLineCollection(numpoints=3)}, framealpha=0, markerfirst=False)
l = axs.legend(handles=[lc,line1, line2], labels=[fr"$\alpha = 1.3, 1.7, 2.3$","Historical", "ODE"], handler_map={lc: HandlerColorLineCollection(numpoints=3)}, loc="upper right", framealpha=0, markerfirst=False)
## m = axs.legend(handles = [historical, ode], labels = ["Historical", "ODE"])
# plt.legend()
# l._legend_box.align = "right"
# renderer = fig.canvas.get_renderer()
# shift = max([t.get_window_extent(renderer).width for t in l.get_texts()])
# for t in l.get_texts():
#     t.set_ha('right') # ha is alias for horizontalalignment
#     t.set_position((shift,0))
# plt.legend()
plt.savefig(f"paper/cornwallarea2.pdf", bbox_inches = "tight")
# add_patch(l)
plt.show()

# plt.savefig(f"paper/chisquaring10{ap}_factor{factor}_resid_3SMALL.pgf", bbox_inches = "tight")



"""

# plt.show()
#plt.plot(years, prop)

# years = np.load("data/Feb/Cornwall/times_cornwallPopTestICcountyAlpha2.0Beta1.1Sigma50.0Deltat0.0001Tmax500.0_0.npy")



years = realpopyears
years = years/2000 + 1000
prop = popprop
print(prop)
print(prop)
#popprop = np.array(prop) #* 0.01


def dxdt(x, t, a, c, s):
    print(a, c, s, end="\r")
    return (1-x)*c*(x**a)*s - c*x*(1-s)*(1-x)**a

def getFunction(x, x0, a, c, s):
    # sol = solve_ivp(dxdt, [wyears[0], wyears[-1]], [wprop[0]], t_eval = wyears, args=(a, c, s))
    sol = odeint(dxdt, x0, years, args=(a, c, s))
    # plt.plot(sol[:,0])
    # sol = np.array([])
    return sol[:,0]

params0 = [2,0.0000000000001,0.4]
vals = getFunction(years, 80, 2, 0.00001, 0.4)
plt.plot(years, vals)
plt.show()

# popt, pcov = curve_fit(getFunction, years, prop, p0 = params0, bounds = ([0., 0., -np.inf, 0.], [np.inf, np.inf, np.inf, 1.]))
#
# print(popt)
sol = solve_ivp(dxdt, t_span = [years[0], years[-1]], y0=[prop[0]], args = (1.4, 0.0001, 0.3))
#
# plt.clf()
plt.plot(years, prop, "ro")
f = getFunction(years, *popt)
plt.plot(years, f, "r-")
print(popt)
perr = np.sqrt(np.diag(pcov))
print(perr)
# plt.plot(sol)
plt.show()
"""