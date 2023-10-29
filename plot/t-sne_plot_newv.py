import os
import sys

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pickle, json

metric = "cosine"
perplexity = 15

tsne = TSNE(n_components=2, perplexity=perplexity,
            learning_rate='auto', metric=metric, early_exaggeration=5)
X_std = []
groups = []
alphas = []

model_name = sys.argv[1]
dir = f"../test_results/ligand_feats/ligands_chembl_indomain_test_{model_name}"
names = []
for f in os.listdir(dir):
    if f.endswith("_y.npy") or f.endswith("json"):
        continue
    name = f[:-4]
    X_std_tmp = np.load(f"{dir}/{name}.npy")
    names.append((name, X_std_tmp.shape[0]))

names = [a[0] for a in sorted(names, key=lambda x:x[1], reverse=True)]
smiles = []
assay_idxes = []
activity = []
for i, name in enumerate(names[:7]):
    X_std_tmp = np.load(f"{dir}/{name}.npy")
    X_std.append(X_std_tmp)
    labels_tmp = np.load(f"{dir}/{name}_y.npy")
    assay_idxes += [i]*len(labels_tmp)
    smiles += [f"{i}___" + x for x in json.load(open(f"{dir}/{name}_smiles.json", "r"))]
    activity += labels_tmp.tolist()
    labels_tmp = labels_tmp - labels_tmp.min()

    alphas.append(labels_tmp / labels_tmp.max())
    groups.append([i]*len(X_std_tmp))

X_std = np.concatenate(X_std, axis=0)
labels = np.concatenate(groups, axis=0)
alphas = np.concatenate(alphas, axis=0)
X_std = (X_std - X_std.mean(axis=0)) / X_std.std(axis=0)
if not os.path.exists(f"X_tsne_{model_name}.npy"):
    X_tsne = tsne.fit_transform(X_std)
    np.save(f"X_tsne_{model_name}.npy", X_tsne)
else:
    X_tsne = np.load(f"X_tsne_{model_name}.npy")

colors = """#4393c3
#33a02c
#f1b6da
#ff7f00
#762a83
#8c510a
#313695""".split("\n")
import plot_settings
from matplotlib.markers import MarkerStyle

style = MarkerStyle('o')

print(names)
names_dict = """CHEMBL3889082: Inhibitors of mutant IDH
CHEMBL2093838: WHO-TDR NTD screening
CHEMBL3705762: Inhibitors of ERK
CHEMBL3889265: Orexin receptor modulators
CHEMBL3887093: Hepatitis C virus infections
CHEMBL3707743: Sodium channels blocked
CHEMBL4418462: Inhibitors of ATG7"""
names_dict = [x.split(": ") for x in names_dict.split("\n")]
names_dict = {x[0]: x[1] for x in names_dict}

plot_legend = False
if plot_legend:
    plt.figure(figsize=(5, 1.2))
    ax = plt.subplot(1, 1, 1)
    for i in range(7):
        ax.bar(x=[0], height=[0], bottom=[0], color=colors[i],
               yerr=[0], label=names_dict[names[i].split('_')[0]])
        # ax.scatter([0], [0], c=colors[unit], alpha=0.0, marker=style, label=names[unit].split('_')[0])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    plt.cla()
    plt.legend(handles, labels, loc="center", scatterpoints=1, ncol=2, frameon=False)
else:
    ax = plot_settings.get_square_axis()
    for i in range(7):
        i_idx = np.nonzero(labels==i)
        X_tsne_i = X_tsne[i_idx]
        ax.scatter(X_tsne_i[:, 0], X_tsne_i[:, 1], c=colors[i], alpha=1.0, marker=style, s=7)


data_show_all = [(pos, smile) for pos, smile in zip(X_tsne, smiles) if -15<=pos[0]<=5 and 30<=pos[1]<=50]

# smiles_show = ['1___COc1ccccc1C(=O)NCCSc1c(-c2ccccc2)[nH]c2ccccc12',
# '0___C[C@H](Nc1nc(Cl)cc(N2C(=O)OC[C@@H]2[C@@H](C)O)n1)c1cc(-c2ccccc2)on1',
# '3___Cc1ccc(C(=O)N2C3CCC2C(COc2cnc4ccccc4n2)C3)c(-n2ccnn2)n1',
# '6___Nc1ncnc2c1c(Sc1c[nH]c3ccccc13)nn2[C@H]1C[C@H](O)[C@@H](COS(N)(=O)=O)O1']
# for pos, smile in zip(X_tsne, smiles):
#     if smile in smiles_show:
#         # circ = plt.Circle((pos[0], pos[1]), radius=10, color='r', alpha=0.5)
#         # ax.add_patch(circ)
#         plt.scatter(pos[0], pos[1], marker='o', edgecolors='r', s=300)
# ax.axis('off')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.xticks([])
plt.yticks([])
if plot_legend:
    plt.savefig('./figs/2.h.t-sne-legend.pdf')
else:
    if "maml" in dir:
        plt.title("MAML", size=16)
        plt.tight_layout()
        plt.savefig('./figs/2.h2.t-sne-maml-onchembl.pdf')
    else:
        plt.title("ActFound", size=16)
        plt.tight_layout()
        plt.savefig('./figs/2.h1.t-sne-ours-onchembl.pdf')
plt.show()
