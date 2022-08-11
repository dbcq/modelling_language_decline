import numpy as np
from matplotlib import pyplot as plt
import matplotlib

# matplotlib.rcParams.update({'font.family': 'serif', "text.usetex": True})
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})  
format = "pgf"
size = (3, 2.75)
# format = "pgf"
xlabel = r"Input"
ylabel = r"Output"
save = True
filename = "sigmoid1"


def sigmoid(m, a=1., b=1.):
    """Modelling the sigmoid function. a is bias torwards feature. b is conformity number."""
    f = m ** (a * b) / (m ** (a * b) + (1 - m ** a) ** b)
    return f

if format == "png":
    matplotlib.rcParams.update({
        "mathtext.fontset": "dejavuserif",
    })

# elif format == "pgf":
#     matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #    "pgf.texsystem": "pdflatex",
    #    'pgf.rcfonts': False,
    # })

fontsize = 11
fig, ax = plt.subplots(figsize = size)
ax.set_xlabel(xlabel, fontsize = fontsize)
ax.set_ylabel(ylabel, fontsize = fontsize)
ax.tick_params(axis='both', labelsize=fontsize)
matplotlib.rc('legend', fontsize=fontsize)
mvals = np.linspace(0, 1, 100)
abvals = np.array([[1.0, 1.0], [1.0, 2.5], [0.8, 2.5]])
yvals = np.zeros((len(mvals), len(abvals)))
labels = ["Neutral", "Conformity", "Conformity + bias"]

for i, ab in enumerate(abvals):
    # yvals[:,i] = sigmoid(mvals, *ab)
    ax.plot(mvals, sigmoid(mvals, *ab))#, label = f"{labels[i]}")

# plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("paper/sigmoid.pdf", bbox_inches = "tight", dpi = 400)

# if format == "png":
#     if save == True:
#         print("saving...")
#         plt.savefig("Feb/{}3.png".format(filename), bbox_inches = "tight")
#     plt.show()
#
# elif format == "pgf":
#     if save == True:
#         print("saving pgf...")
#         plt.savefig("Report/{}3.pgf".format(filename), bbox_inches = "tight")


