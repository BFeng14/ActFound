
import os
import sys
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
from matplotlib.markers import MarkerStyle

sys.path.append(os.path.join(sys.path[0], '../'))
import plot_settings
import warnings
warnings.filterwarnings('ignore')

model_name = sys.argv[1]
if model_name == "metadelta":
    X_tsne = np.load("./t-sne/t-sne-ddg.npy")
elif model_name == "maml":
    X_tsne = np.load("./t-sne/t-sne-dg.npy")
else:
    exit()

labels = np.load("./t-sne/t-sne-labels.npy")
data_labels = ["CHEMBL3707743", "CHEMBL3887093",
               "CHEMBL3705762", "CHEMBL3889082",
               "CHEMBL3889265", "CHEMBL2093838"]

ax = plot_settings.get_wider_axis()
style = MarkerStyle('o')

ax.set_xlabel('tSNE1', loc='left')
ax.set_ylabel('tSNE2', loc='bottom')

ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, alpha=0.5, marker=style)
plt.xticks([])
plt.yticks([])

if model_name == "metadelta":
    plt.savefig('./figs/2.g1.t-sne-Meta-DDG.pdf')
elif model_name == "maml":
    plt.savefig('./figs/2.g2.t-sne-maml.pdf')
plt.show()