import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)


import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from matplotlib_scalebar.scalebar import ScaleBar

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
path = 'data/Mar/walesICbook5Alpha1.0Beta1.1SigmavarFactor1Deltat0.0004Tmax500.0_1_'
ksa1 = [0, 5000, 250000]
kssconst = [4000, 9500, 25000]
kssvar = [6500, 28000, 55000]
ksa1_new = [2500, 54500, 81500]
filenames = []
for k in ksa1_new:
    # nametemplate = f"walesICbook5Alpha1.0Beta1.1Sigma25Deltat0.0004Tmax500.0MEMORY_2_{k}.npy"
    nametemplate = f"walesICbook5Alpha1.0Beta1.1SigmavarFactor1Deltat0.0004Tmax500.0MEMORY_1_{k}.npy"
    filenames.append(nametemplate)

# filenames = ["cornwallPopGaussian10ICriverAlpha1.5Beta1.1SigmavarDeltat0.0001Tmax500.0MEMORY_1_234000.npy"]*3
includedRegion = np.load("countries/Wales/wales_plus_mask.npy").astype(bool)[:586, 50:540]

wales = np.load("countries/Wales/wales_fullsize_country_mask.npy")[:586, 50:540]

populationDensity = np.load("countries/Wales/smoothed_PopDistnew5_2.npy")
populationDensity = populationDensity[:586, 50:540]
# Single
# for i, k in enumerate(ks):
for i, filename in enumerate(filenames):
    # plt.cla()
    if i == 0:
        s = fig.get_size_inches()
        print(s)
        fig.set_size_inches(float(s[0])*0.85, float(s[1])*0.6*0.85/0.85)

        # scalebar = AnchoredSizeBar(axs[i].transData,
        #                            100, 'asdfjgkhasld', 'lower center',
        #                            pad=0.3,
        #                            color='black',
        #                            frameon=False,
        #                            size_vertical=2)

    # filename = f"mUniformforbigfigure{k}.npy"
    savename = f"mWales{i}.pdf"
    filepath = os.path.join(path, filename)
    savepath = os.path.join(path, savename)
    m = np.load(filepath)
    m = m[:586, 50:540]
    m[~wales] = np.nan
    # populationDensity = np.ones((m.shape[0], m.shape[1]))

    # im = axs[i].imshow(populationDensity, origin = "lower", vmin = 0, vmax = 21)
    # vmin, vmax = im.get_clim()
    # print(vmin, vmax)
    im = axs[i].imshow(m, origin='lower', cmap=plt.cm.jet, vmin = 0.0, vmax = 1.0)  # Here make an AxesImage rather than contour
    # axs[i].imshow(wales, alpha = 0.2, origin = "lower")
    if isogloss == "isogloss":
        if i == 1:
            isogloss_loc = np.where(np.abs(m - 0.5) < 0.15)
        else:
            isogloss_loc = np.where(np.abs(m - 0.5) < 0.15)
        isogloss_array = np.zeros_like(m)
        isogloss_array.fill(np.nan)
        isogloss_array[isogloss_loc] = 255
        axs[i].imshow(isogloss_array, cmap="binary", origin="lower", vmin=0., vmax=255)

    q = axs[i].get_position()
    print(q)

    # ax2pos.x1 + 0.05, ax2pos.y0, .05, ax2pos.height
    #

    # scalebar = ScaleBar(0.4, "km", frameon=True, location="center left")  # 1 pixel = 0.2 meter
    # axs[0].add_artist(scalebar)

    if i == 2:
        div = make_axes_locatable(ax)
        # cax = div.append_axes('right', '5%', '5%')
        cax = fig.add_axes([q.x1+0.02, q.y0, 0.015, q.height])
        cb = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, ticks=[0.0, 0.5, 1.0])
        cb.set_label("Frequency", rotation=270, labelpad=15)

        scalebar = AnchoredSizeBar(axs[i].transData,
                                   100, '40 km', 'center left',
                                   pad=0.3,
                                   color='black',
                                   frameon=False,
                                   size_vertical=1)
        axs[i].add_artist(scalebar)



    fig.tight_layout()
    plt.subplots_adjust(wspace = 0.22)
# plt.savefig("Report/walesa1.7sconst.pgf", bbox_inches = "tight", dpi = 500)
# plt.savefig("Trevs/walescountry1.7var.pgf", bbox_inches = "tight", dpi = 500)
plt.savefig("paper/wales1.0.pdf", bbox_inches = "tight")#, dpi = 500)
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
