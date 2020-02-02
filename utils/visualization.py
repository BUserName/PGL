import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects


import seaborn as sns
import torch
import numpy as np

import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)

def visualize_TSNE(feat, label, num_class, args, split):

    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
    label = torch.cat(label).numpy()
    label = (label > num_class-2).astype(np.int).reshape(-1)
    label = label + 1
    ind = np.argsort(label)
    split_idx = np.array(torch.cat(split)).reshape(-1)

    target_idx = np.where(split_idx == 1)[0]
    source_idx = np.where(split_idx == 0)[0]


    feat = torch.cat(feat)
    dim = feat.size()
    feat = feat.view(dim[0]*dim[1], dim[2])
    target_feat = feat[target_idx]
    source_feat = feat[source_idx]



    # dim = target_feat.size()
    # target_feat = target_feat.view(dim[0]*dim[1], dim[2])
    #
    # src_dim = source_feat.size()
    # source_feat = source_feat.view(src_dim[0]*src_dim[1], src_dim[2])
    # source_feat_select = source_feat[np.random.choice(source_feat.size(0), int(source_feat.size(0)/2))]
    src_label = np.full([source_feat.shape[0]], 0)


    X = np.vstack(target_feat[ind])
    X = np.concatenate([source_feat,X])
    y = np.hstack(label[ind])
    y = np.concatenate([src_label, y])

    digits_proj = TSNE(random_state=2020).fit_transform(X)

    # We choose a color palette with seaborn.
    flatui = ["#1d8bff",  "#ff5e1d","#c5c5c5"]
    palette = np.array(sns.color_palette(flatui, 3))
    label = ['Source', 'Target Known','Target Unknown']
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(3):
        idx = np.where(y==i)
        sc = ax.scatter(digits_proj[idx, 0], digits_proj[idx, 1], lw=0, s=15,
                        c=palette[i], label=label[i])

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5, markerscale=4)
    txts = []
    # We add the labels for each digit.
    # txts = ["back_pack", "bike", "bike_helmet", "bookcase", "bottle",
    #                        "calculator", "desk_chair","desk_lamp","desktop_computer","file_cabinet","unk"]
    # for i in range(len(txts)):
    #     # Position of each label.
    #     xtext, ytext = np.median(digits_proj[y == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, txts[i], fontsize=12)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)

    return f, ax, sc, txts


