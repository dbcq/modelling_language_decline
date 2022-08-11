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
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)

                                                  
                                                  
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
# # realyears = np.array([1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1800])
# realareayears = np.array([1200, 1300, 1400, 1450, 1500, 1600, 1650, 1700, 1750, 1800])
# realpopyears = np.array([1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1800])
# # realareas = np.array([np.nan,3270,np.nan,2780,np.nan,2360,2360,1890,np.nan,1400,910,530,160,0])
# realareas = np.array([3270, 2780,2360,2360,1890,1400,910,530,160,0])
# atot = 3563
# # atot = 3270
# areaprop = realareas/atot * 100
# # totalPops = np.array([21, 28,35,43,52,48,55,62,69,76,84,93,106,192])*1000
# totalPops = np.array([35, 43,52,48,55,62,69,76,84,93,106,192])*1000
# # cornishSpeakers = np.array([20,26,30,34,38,32,34,33,33,30,22,14,5,0])*1000
# cornishSpeakers = np.array([30, 34,38,32,34,33,33,30,22,14,5, 0])*1000
# popprop = (cornishSpeakers/totalPops) * 100
# p0 = prop[0]
realpopyears = [1901, 1911, 1921, 1931, 1951, 1961, 1971, 2001]
realPop = [49.9, 43.5, 37.1, 36.8, 28.9, 26.0, 20.9, 20.5]
popprop = np.array(realPop)/100

# def fitfunc(t, a, c, s):
#     def dxdt(x, t, a, c, s):
#         return (1-x)*c*(x**a)*s - c*x*(1-x)**a * (1-s)
#
#     x_0 = proportion[0]
#     x_sol = odeint(dxdt, x_0, t, args = (a, c, s))d
#     return x_sol
#
# paramfit, paramcov = curve_fit(fitfunc, years, proportion)
# print(paramfit)
# print(np.sqrt(np.diag(paramcov)))
#
# tfit = np.linspace(1100, 1800)
# fit = fitfunc(tfit, *paramfit)


# wyears = [1901, 1911, 1921, 1931, 1951, 1961, 1971, 1981]
# wprop = [13, 9.6, 6.4, 6.0, 3.5, 3.4, 2.1, 2.7]
# years = np.load("data/Feb/Cornwall/times_cornwallPopTestICcountyAlpha1.2Beta1.1Sigma50.0Deltat0.0001Tmax500.0_4.npy")
# prop = np.load("data/Feb/Cornwall/pop1_cornwallPopTestICcountyAlpha1.2Beta1.1Sigma50.0Deltat0.0001Tmax500.0_4.npy")
# prop = np.array(prop) * 0.01

sigma_smooth = 5
walesMask = np.load("countries/Wales/wales_fullsize_country_mask.npy").astype(bool)
initialMask = np.load("countries/Wales/wales_initial_1850.npy").astype(bool)
walesPopulation = np.load(f"countries/Wales/smoothed_PopDistnew5_2.npy")
walesPopulation[~walesMask] = np.nan
popTot = np.nansum(walesPopulation[walesMask])
areaTot = np.nansum(walesMask.astype(int))
print(popTot)

initialProp = np.nansum(walesPopulation[initialMask])/np.nansum(walesPopulation[walesMask])
print("initial:", initialProp)

# alphas = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]
alphas = [1.2, 1.7, 2.3]
ap = "p"
sigma = 25
endyear = 4087#3479 #3508
# numbers = ["all", "all", "all", 0]
fig, axs = plt.subplots(1, 1, figsize = (3.3, 3.1))#, gridspec_kw={'height_ratios': [3, 1]})
ax2 = plt.axes([0,0,1,1])
for number, alpha in enumerate(alphas):
    # name_template = f"walesICbook5Alpha{alpha}Beta1.1Sigma25Deltat0.0004Tmax500.0"
    name_template = f"walesICbook5Alpha{alpha}Beta1.1SigmavarFactor1Deltat0.0004Tmax500.0"
    # num = numbers[number]
    num = 1#2#1
    path = "data/Mar"
    timespath = os.path.join(path, f"times_Gaussian_{name_template}_{num}.npy")
    times = np.load(timespath)
    if ap == "p":
        poppath = os.path.join(path, f"pop1_Gaussian_2_{name_template}_{num}.npy")
        pops = np.load(poppath)

        prop = pops/popTot
        propinitial = prop[0]
        # prop = prop*popprop[0]/propinitial
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
        # chisqlist.append(chisq)

        MSE = 0
        N = popprop.shape[0]
        for i, year in enumerate(realpopyears):
            index = (np.abs(years - year)).argmin()
            # closestYear = years[index]
            Ei = prop[index]
            # print("Oi :", Ei)
            Oi = popprop[i]
            # print("Ei:", Oi)
            MSEcontrib = (Oi - Ei) ** 2
            MSE += MSEcontrib
        MSE = MSE / N

    elif ap == "a":
        areapath = os.path.join(path, f"area1_Gaussian_2_{name_template}_{num}.npy")
        areas = np.load(areapath)
        areas = np.squeeze(areas)
        prop = areas/areaTot
        # prop = areas/np.max(areas) * 100
        lastIndex = np.argwhere(prop == np.min(prop))[0]
        prop = prop[np.argmax(areas):lastIndex[0]+1]
        years = np.linspace(1850, 2150, lastIndex[0]+1-np.argmax(areas))
        times = times[np.argmax(areas):lastIndex[0]+1]
        #
        # chisq = 0
        # for i, year in enumerate(realareayears):
        #     index = (np.abs(years - year)).argmin()
        #     closestYear = years[index]
        #     Ei = prop[index]
        #     print("Oi :", Ei)
        #     Oi = areaprop[i]
        #     print("Ei:", Oi)
        #     chisqContrib = ((Oi - Ei) ** 2) / Ei
        #     chisq += chisqContrib






    a = axs.plot(years, prop)#, label = MSE)
    a = ax2.plot(years, prop)#, label = MSE)

#realareas/3563 *100

def dxdt(x, t, a, c, s):
    print(a, c, s, end="\r")
    return (1-x)*c*(x**a)*s - c*x*(1-s)*(1-x)**a

def getFunction(x, x0, a, c, s):
    # sol = solve_ivp(dxdt, [wyears[0], wyears[-1]], [wprop[0]], t_eval = wyears, args=(a, c, s))
    sol = odeint(dxdt, x0, realpopyears, args=(a, c, s))
    # sol = np.array([])
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
    # ODEresids.append(Ei-Oi)
    chisq += chisqContrib

MSE = 0

for i, year in enumerate(realpopyears):
    Ei = f[i]
    print("Oi :", Ei)
    Oi = popprop[i]
    print("Ei:", Oi)
    MSEcontrib = (Oi - Ei) ** 2
    # ODEresids.append(Ei-Oi)
    MSE += MSEcontrib
MSE = MSE/N
ax2.plot(realpopyears, f, color="dimgrey", linestyle="--",linewidth = 2)#,label = fr"$\chi^2 = {MSE}")
axs.plot(realpopyears, f, color="dimgrey", linestyle="--", linewidth = 2)#,label = fr"$\chi^2 = {MSE}")
axs.plot(realpopyears, popprop, marker = ".", color = "r",linestyle = "")#, label = "Historical data")
ax2.plot(realpopyears, popprop, marker = ".", color = "r",linestyle = "")#, label = "Historical data")
print("chi^2 = ", chisq)


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
lc.set_array(np.array(realpopyears))

ax2.add_collection(lc)
#
line1 = Line2D([0], [0], color = 'red', marker='.', linestyle='')
line2 = Line2D([0], [0], color = 'dimgrey', linestyle='--')
#
#
l = ax2.legend(handles=[lc, line1], labels =[fr"$\alpha = 1.3, 1.7, 2.3$", "Historical"], handler_map = {lc: HandlerColorLineCollection(numpoints=3)}, framealpha=0, markerfirst=False)
l = ax2.legend(handles=[lc,line1, line2], labels=[fr"$\alpha = 1.3, 1.7, 2.3$","Historical", "ODE"], handler_map={lc: HandlerColorLineCollection(numpoints=3)}, loc="upper right", framealpha=0, markerfirst=False)
# m = axs.legend(handles = [historical, ode], labels = ["Historical", "ODE"])

# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
# ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
# SIGMA VAR
# ip = InsetPosition(axs, [0.3,0.4,0.7,0.6])
# ax2.set_axes_locator(ip)
# xmin = 1870
# xmax = 2100
# ymin = 0.13
# ymax = 0.7# + (xmax-xmin)
# extent = (xmin, xmax, ymin, ymax)
# ax2.set_xlim(xmin, xmax)
# ax2.set_ylim(ymin, ymax)
# # ax2.imshow(popOrig[ymin:ymax, xmin:xmax], extent = extent, origin = "lower")
# ax2.plot(realpopyears, f, "k-", linewidth = 2,label = fr"$\chi^2 = {chisq}")
# # ax2.axes.xaxis.set_visible(False)
# # ax2.axes.yaxis.set_visible(False)
# # ax2.axis("off")
# # Mark the region corresponding to the inset axes on ax1 and draw lines
# # in grey linking the two axes.
# mark_inset(axs, ax2, loc1=2, loc2=4, fc="none", ec='0.5', color = "k")

#SIGMA CONST
ip = InsetPosition(axs, [0.3,0.4,0.7,0.6])
ax2.set_axes_locator(ip)
xmin = 1870
xmax = 2100
ymin = 0.13
ymax = 0.7# + (xmax-xmin)
extent = (xmin, xmax, ymin, ymax)
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)
# ax2.imshow(popOrig[ymin:ymax, xmin:xmax], extent = extent, origin = "lower")

# ax2.axes.xaxis.set_visible(False)
# ax2.axes.yaxis.set_visible(False)
# ax2.axis("off")
# Mark the region corresponding to the inset axes on ax1 and draw lines
# in grey linking the two axes.
# mark_inset(axs, ax2, loc1=2, loc2=4, fc="none", ec='0', color = "k")



# straightLine = -91.78/600 * (realpopyears-1200) + 91.78
# chisqStraightLine = np.sum([(popprop[i]-straightLine[i])**2 / straightLine[i] for i in range(len(straightLine))])
#
# axs.plot(realpopyears, straightLine, color = "k", marker = "", linestyle = "-", alpha = 0.5, label = fr"$\chi^2 = {int(round(chisq))}$")

size = 11
axs.set_xlabel("Year", fontsize = size)
axs.set_ylabel("Population proportion", fontsize = size)
axs.tick_params(labelsize=size)
axs.xaxis.offsetText.set_fontsize(size)
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
# plt.legend(fontsize = size)
# plt.legend()
# plt.ylim(0, 100)
# plt.savefig("Report/chisquaring.png", bbox_inches = "tight")
plt.savefig("paper/walespopulationsigmavar.pdf", bbox_inches = "tight")



plt.show()
#plt.plot(years, prop)

# years = np.load("data/Feb/Cornwall/times_cornwallPopTestICcountyAlpha2.0Beta1.1Sigma50.0Deltat0.0001Tmax500.0_0.npy")
"""
#years = years/2000 + 1000
print(prop)
prop = np.array(prop) #* 0.01


def dxdt(x, t, a, c, s):
    print(a, c, s, end="\r")
    return (1-x)*c*(x**a)*s - c*x*(1-s)*(1-x)**a

def getFunction(x, x0, a, c, s):
    # sol = solve_ivp(dxdt, [wyears[0], wyears[-1]], [wprop[0]], t_eval = wyears, args=(a, c, s))
    sol = odeint(dxdt, x0, years, args=(a, c, s))
    # sol = np.array([])
    return sol[:,0]

params0 = [2,0.0000000000001,0.4]

popt, pcov = curve_fit(getFunction, years, prop, bounds = ([0., 0., -np.inf, 0.], [np.inf, np.inf, np.inf, 1.]))

# print(popt)
# sol = solve_ivp(dxdt, t_span = [years[0], years[-1]], y0=[0.9], args = (1.3, 0.0001, 0.3))

plt.clf()
plt.plot(years, prop, "ro")
f = getFunction(years, *popt)
plt.plot(years, f, "r-")
print(popt)
perr = np.sqrt(np.diag(pcov))
print(perr)
# plt.plot(sol)
plt.show()
"""