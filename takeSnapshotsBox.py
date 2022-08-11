import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import os

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})  
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
# matplotlib.use("pgf")

format = "png"
colorbar = "colorbar"
save = True
n = 10
fig, axs = plt.subplots(1, 3)
for ax in axs:
    ax.set_aspect('equal')
    ax.axis("off")
isogloss = "isogloss"
path = "data/Mar/"
ks = [10, 100, 300]
mAll = np.load("data/Mar/alpha1.05beta1.1sigma1.0tmax1000popDistminusgap9exp20_80xlongMEMORY_every100.npy")
# mAll = np.load("data/Mar/alpha1.05beta1.1sigmavariabletmax1000popDistminusgap9exp20_80xlongMEMORY_every100.npy")

# filenames = ["cornwallPopGaussian10ICriverAlpha1.5Beta1.1SigmavarDeltat0.0001Tmax500.0MEMORY_1_234000.npy"]*3
# includedRegion = np.load("countries/Cornwall_data/cornwall_mask_w_border_rightway.npy").astype(bool)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1:
        n = cmap.N
    new_cmap = mcolors.LinearSegmentedColormap.from_list(
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval),
         cmap(np.linspace(minval, maxval, n)))
    return new_cmap

minColor = 0.00
maxColor = 0.55
viridis = truncate_colormap(plt.get_cmap("viridis"), minColor, maxColor)



# Single
# for i, k in enumerate(ks):
for i, k in enumerate(ks):
    # plt.cla()
    if i == 0:
        s = fig.get_size_inches()
        print(s)
        fig.set_size_inches(float(s[0])*0.85, float(s[1])*0.9)
    # filename = f"mUniformforbigfigure{k}.npy"
    # savename = f"blocktest{k}.png"
    # filepath = os.path.join(path, filename)
    # savepath = os.path.join(path, savename)
    # m = np.load(filepath)
    # populationDensity = np.ones((m.shape[0], m.shape[1]))
    populationDensity = np.load("data/Mar/blocktestpopdist.npy")
    populationDensity = populationDensity[1:-1,1:-40]
    im = axs[i].imshow(populationDensity, cmap = viridis, origin = "lower", vmin = 0, vmax =10)

    # vmin, vmax = im.get_clim()
    # print(vmin, vmax)
    m = mAll[k,:,:-40]
    # im = axs[i].imshow(m, origin='lower', cmap=plt.cm.jet, vmin = 0.0, vmax = 1.0)  # Here make an AxesImage rather than contour

    if isogloss == "isogloss":
        if i == 2:
            isogloss_loc = np.where(np.abs(m - 0.5) < 0.35)
        else:
            isogloss_loc = np.where(np.abs(m - 0.5) < 0.2)
        isogloss_array = np.zeros_like(m)
        isogloss_array.fill(np.nan)
        isogloss_array[isogloss_loc] = 255
        axs[i].imshow(isogloss_array, cmap="binary", origin="lower", vmin=0., vmax=255)

    q = axs[i].get_position()
    print(q)

    # ax2pos.x1 + 0.05, ax2pos.y0, .05, ax2pos.height
    #

    if i == 2:
        div = make_axes_locatable(ax)
        # cax = div.append_axes('right', '5%', '5%')
        cax = fig.add_axes([q.x1+0.02, q.y0, 0.015, q.height])
        cb = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04)#, ticks=[0.0, 0.5, 1.0])
        cb.set_label("Population density", rotation=270, labelpad=15)
        # cb.set_clim(0, 10)


    fig.tight_layout()
print(np.max(populationDensity))
plt.savefig("paper/blocktest0_1.pdf", bbox_inches = "tight", dpi = 500)
plt.show()
#
# levels = np.linspace(0, 1, 11)
#
# cb = plt.colorbar(contourplot, ticks = [0.0, 0.5, 1.0])
# cb.ax.tick_params(labelsize=26)



# if format == "png":
#     if save == True:
#         print("saving...")
#         plt.savefig("graphs/Interim/{}.png".format(filename), bbox_inches = "tight")
#         plt.show()
#     plt.show()
