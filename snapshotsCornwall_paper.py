import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

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
path = "data/Mar/cornwallPopGaussian10ICriverAlpha1.5Beta1.1SigmavarFactor1Deltat0.0004Tmax500.0_0_"
ksa1 = [0, 100000, 1000000]
kss25 = [2000, 20000, 60000]
kss50 = [2000, 100000, 220000]
filenames = []
for k in kss50:
    # nametemplate = f"cornwallPopGaussian10ICriverAlpha1.5Beta1.1Sigma50Deltat0.0001Tmax500.0MEMORY_0_{k}.npy"
    nametemplate = f"cornwallPopGaussian10ICriverAlpha1.5Beta1.1SigmavarFactor1Deltat0.0004Tmax500.0MEMORY_0_{k}.npy"
    filenames.append(nametemplate)

# filenames = ["cornwallPopGaussian10ICriverAlpha1.5Beta1.1SigmavarDeltat0.0001Tmax500.0MEMORY_1_234000.npy"]*3
includedRegion = np.load("countries/Cornwall_data/cornwall_mask_w_border_rightway.npy").astype(bool)

# Single
# for i, k in enumerate(ks):
for i, filename in enumerate(filenames):
    # plt.cla()
    if i == 0:
        s = fig.get_size_inches()
        print(s)
        # fig.set_size_inches(float(s[0])*0.85, float(s[1])*0.9)
        fig.set_size_inches(float(s[0])*0.9, float(s[1])*0.9)
    # filename = f"mUniformforbigfigure{k}.npy"
    savename = f"mCornwalltest{i}.png"
    filepath = os.path.join(path, filename)
    savepath = os.path.join(path, savename)
    m = np.load(filepath)
    populationDensity = np.ones((m.shape[0], m.shape[1]))
    populationDensity = np.load("Report/simpleExamplePopulationDensity.npy")
    # im = axs[i].imshow(populationDensity, origin = "lower", vmin = 0, vmax = 21)
    # vmin, vmax = im.get_clim()
    # print(vmin, vmax)
    im = axs[i].imshow(m, origin='lower', cmap=plt.cm.jet, vmin = 0.0, vmax = 1.0)  # Here make an AxesImage rather than contour

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

    if i == 2:
        div = make_axes_locatable(ax)
        # cax = div.append_axes('right', '5%', '5%')
        cax = fig.add_axes([q.x1+0.02, q.y0, 0.015, q.height])
        cb = fig.colorbar(im, cax=cax, fraction=0.046, pad=0.04, ticks=[0.0, 0.5, 1.0])
        cb.set_label("Frequency", rotation=270, labelpad=15)

        scalebar = AnchoredSizeBar(axs[i].transData,
                                   100, '20 km', 'lower right',
                                   pad=0.3,
                                   color='black',
                                   frameon=False,
                                   size_vertical=1)
        axs[i].add_artist(scalebar)
        # fontproperties=fontprops)


    fig.tight_layout()

# plt.subplots_adjust(wspace=0.2)

plt.savefig("paper/cornwall1.5.pdf", bbox_inches = "tight", dpi = 500)
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
