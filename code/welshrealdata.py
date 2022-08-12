import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


d = Image.open("wales_current_welsh_usage.tif")
mask = Image.open("wales_current_usage_mask.tif")
d = np.array(d) * 1/100
mask = np.array(mask).astype(bool)
d = np.flip(d, axis=0)
mask = np.flip(mask, axis=0)
d[~mask] = np.nan

fig, ax = plt.subplots()

plt.imshow(d, origin="lower")
cb = plt.colorbar()
cb.set_label("Proportion able to speak Welsh", rotation=270, labelpad=15)

ax.axis("off")
s = fig.get_size_inches()
fig.set_size_inches(float(s[0])*0.6, float(s[1])*0.6)
plt.savefig("wales_current_usage.pdf", bbox_inches="tight")
plt.show()
