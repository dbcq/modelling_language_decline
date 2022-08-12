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


format = "png"
colorbar = "colorbar"
save = True
n = 10
fig, axs = plt.subplots(1, 3)
for ax in axs:
    ax.set_aspect('equal')
    ax.axis("off")
isogloss = "isogloss"
path = "data"

# ks10 = [2000, 100000, 220000]    # alpha = 1.0
ks15 = [54000, 106000, 228000]    # alpha = 1.5
filenames = []
for k in ks15:
    # nametemplate = f"cornwallPopGaussian10ICriverAlpha1.0Beta1.1SigmavarFactor1Deltat0.0004Tmax500.0MEMORY_0_{k}.npy"
    nametemplate = f"cornwallPopGaussian10ICriverAlpha1.5Beta1.1SigmavarFactor1Deltat0.0004Tmax500.0MEMORY_0_{k}.npy"
    filenames.append(nametemplate)

includedRegion = np.load("cornwall_mask_w_border_rightway.npy").astype(bool)

for i, filename in enumerate(filenames):
    if i == 0:
        s = fig.get_size_inches()
        print(s)
        fig.set_size_inches(float(s[0])*0.9, float(s[1])*0.9)
    savename = f"mCornwalltest{i}.png"
    filepath = os.path.join(path, filename)
    savepath = os.path.join(path, savename)
    m = np.load(filepath)
    populationDensity = np.ones((m.shape[0], m.shape[1]))
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

    if i == 2:
        div = make_axes_locatable(ax)
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


    fig.tight_layout()

plt.savefig("cornwall1.5.pdf", bbox_inches = "tight", dpi = 500)
plt.show()
