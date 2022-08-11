import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib
# matplotlib.use("pgf")

matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

fig,ax = plt.subplots()
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
# fontprops = fm.FontProperties(size=12)

d = Image.open("countries/Wales/waleswelshagain2.tif")
mask = Image.open("countries/Wales/waleswelshmask.tif")
d = np.array(d) * 1/100
mask = np.array(mask).astype(bool)
d = np.flip(d, axis = 0)
mask = np.flip(mask, axis = 0)

d[~mask] = np.nan
plt.imshow(d, origin = "lower")
cb = plt.colorbar()
cb.set_label("Proportion able to speak Welsh", rotation=270, labelpad=15)

ax.axis("off")
s = fig.get_size_inches()
print(s)
fig.set_size_inches(float(s[0])*0.6, float(s[1])*0.6)
plt.savefig("paper/walescurrentusagesmaller.pdf", bbox_inches = "tight")
# plt.imshow()
# plt.show()