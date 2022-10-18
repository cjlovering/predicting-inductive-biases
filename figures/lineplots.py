import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
import numpy as np
import glob
import itertools

from collections import ChainMap
from collections import Counter
from collections import defaultdict
from scipy.stats import zscore

import scipy.stats as stats

import matplotlib.colors as colors
import matplotlib.cm as cm
from sklearn.metrics import ndcg_score

from palettable.colorbrewer.diverging import RdBu_4 as RedBlue


def plot(MEASURE, NAME, Y, MODEL, Y_label, _MIN, _MAX, results, probing):
    data = results.copy()
    data = data[data.model == MODEL]
    data = data[~data.prop.str.contains("toy")]

    data[MEASURE] = data[MEASURE].apply(lambda x: round(x, 4))
    data = data.sort_values(by=[MEASURE])

    # This is down to handle the log space with a zero value.
    data["rate"][data.rate == 0] += 1e-4

    NUM_PROPS = len(data.prop.unique())
    filled_markers = (
        "<",
        "X",
        "o",
        ">",
        "8",
        "s",
        "p",
        "X",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
        "o",
        "v",
        "^",
        "<",
        ">",
        "8",
        "s",
    )
    filled_markers = filled_markers[:NUM_PROPS]

    cmap = RedBlue.get_mpl_colormap()
    norm = colors.DivergingNorm(vmin=_MIN, vcenter=1.0, vmax=_MAX)

    f, axes = plt.subplots(
        1,
        2,
        figsize=(8, 2.5),
        gridspec_kw={"width_ratios": [2, 5]},
        constrained_layout=True,
    )

    ################# barplot (c)
    probing_full_data = probing.copy()
    probing_full_data = probing_full_data[probing_full_data.model == MODEL]
    probing_full_data = probing_full_data.sort_values(by=[MEASURE])
    probing_full_data[MEASURE] = probing_full_data[MEASURE].apply(lambda x: round(x, 4))
    pal = dict(
        ChainMap(
            *data.apply(lambda x: {x["prop"]: cmap(norm(x[MEASURE]))}, axis=1).to_list()
        )
    )
    ax = sns.barplot(
        y="prop",
        x=MEASURE,
        data=probing_full_data,
        palette=pal,
        order=data.prop.unique(),
        ax=axes[0],
        orient="h",
    )
    ax.set_yticks([])
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    ax.axvline(1, ls="--", c="#e7298a")
    ax.set_ylabel(r"$s, t$ feature pairs")
    ax.set_xlabel("Relative Extractability of Target Feature\n ( MDL($s$)/MDL($t$) )")
    sns.scatterplot(
        x=[-0.3 for _ in filled_markers],
        y=[i for i in range(len(filled_markers))],
        style=[i for i in range(len(filled_markers))],
        hue=[i for i in range(len(filled_markers))],
        markers=filled_markers,
        palette=[pal[prop] for prop in data.prop.unique()],
        ax=ax,
    )
    ax.get_legend().remove()

    ################# lineplot (a)
    _data = data
    _mean = _data.groupby(["rate", "prop"]).mean().reset_index()

    Y = "weak-error"
    #     Y = "test_f_score"

    ax = sns.lineplot(
        x="rate",
        y=Y,
        hue=MEASURE,
        dashes=False,
        markers=filled_markers,
        style="prop",
        linewidth=1.7,
        data=_mean,
        alpha=1,
        palette=cmap,
        hue_norm=norm,
        ax=axes[1],
        hue_order=data.prop.unique(),
        style_order=data.prop.unique(),
    )

    ax.get_legend().remove()

    # sort or reorder the labels and handles
    ax.set_xscale("log")
    ax.set_xticks([0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
    ax.set_xticklabels([0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
    ax.set_xlabel("Evidence from Spurious-only Examples\n ($s$-only example rate)")
    ax.set_ylabel("Use of Spurious Feature\n ($s$-only error)")

    model_name = MODEL.replace("-", "_").split("_")[0]
    plt.title(model_name)
    plt.savefig(f"figures/{model_name}.pdf", transparent=True)
    plt.close()


def plot_singles(probing, results):
    Y = "weak-error"
    Y_label = "Use of Spurious Feature"

    _probing = probing.copy()
    _mean = _probing.groupby(["prop"]).mean().reset_index()

    _MIN, _MAX = _mean["weak%strong-mdl"].min(), _mean["weak%strong-mdl"].max()
    plot("weak%strong-mdl", "acc", Y, "t5-base", Y_label, _MIN, _MAX, results, probing)
    plot(
        "weak%strong-mdl",
        "acc",
        Y,
        "bert-base-uncased",
        Y_label,
        _MIN,
        _MAX,
        results,
        probing,
    )
    plot(
        "weak%strong-mdl", "acc", Y, "lstm-glove", Y_label, _MIN, _MAX, results, probing
    )
    plot("weak%strong-mdl", "acc", Y, "gpt2", Y_label, _MIN, _MAX, results, probing)
    plot(
        "weak%strong-mdl",
        "acc",
        Y,
        "roberta-base",
        Y_label,
        _MIN,
        _MAX,
        results,
        probing,
    )


# rename for "readability"
def plot_full(MEASURE, NAME, Y, MODEL, Y_label, _MIN, _MAX, results, probing):
    data = results.copy()
    data = data[data.model == MODEL]

    data[MEASURE] = data[MEASURE].apply(lambda x: round(x, 4))
    data = data.sort_values(by=[MEASURE])

    # This is here to handle the log space with a zero value.
    data["rate"][data.rate == 0] += 1e-4

    NUM_PROPS = len(data.prop.unique())
    filled_markers = (
        "<",
        "X",
        "o",
        ">",
        "8",
        "s",
        "p",
        "X",
        "h",
        "H",
        "D",
        "d",
        "P",
        "X",
        "o",
        "v",
        "^",
        "<",
        ">",
        "8",
        "s",
    )
    filled_markers = filled_markers[:NUM_PROPS]

    cmap = RedBlue.get_mpl_colormap()

    norm = colors.DivergingNorm(vmin=_MIN, vcenter=1.0, vmax=_MAX)

    f, axes = plt.subplots(
        4,
        2,
        figsize=(8, 10),
        gridspec_kw={"width_ratios": [2, 5]},
        constrained_layout=True,
    )

    axes[1][0].remove()
    axes[2][0].remove()
    axes[3][0].remove()

    ################# barplot (c)
    probing_full_data = probing.copy()
    probing_full_data = probing_full_data[probing_full_data.model == MODEL]
    probing_full_data = probing_full_data.sort_values(by=[MEASURE])

    probing_full_data[MEASURE] = probing_full_data[MEASURE].apply(lambda x: round(x, 4))
    pal = dict(
        ChainMap(
            *data.apply(lambda x: {x["prop"]: cmap(norm(x[MEASURE]))}, axis=1).to_list()
        )
    )
    ax = sns.barplot(
        y="prop",
        x=MEASURE,
        data=probing_full_data,
        palette=pal,
        order=data.prop.unique(),
        ax=axes[0][0],
        orient="h",
    )

    ax.set_yticks([])
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    ax.axvline(1, ls="--", c="#e7298a")
    ax.set_ylabel(r"$s, t$ feature pairs")
    ax.set_xlabel("Relative Extractability of Target Feature\n ( MDL($s$)/MDL($t$) )")
    sns.scatterplot(
        x=[-0.3 for _ in filled_markers],
        y=[i for i in range(len(filled_markers))],
        style=[i for i in range(len(filled_markers))],
        hue=[i for i in range(len(filled_markers))],
        markers=filled_markers,
        palette=[pal[prop] for prop in data.prop.unique()],
        ax=ax,
    )
    ax.get_legend().remove()

    ################# lineplot (a)
    _data = data
    _mean = _data.groupby(["rate", "prop"]).mean().reset_index()

    Y = "both-error"
    ax = sns.lineplot(
        x="rate",
        y=Y,
        hue=MEASURE,
        dashes=False,
        markers=filled_markers,
        style="prop",
        linewidth=1.7,
        data=_mean,
        alpha=1,
        palette=cmap,
        hue_norm=norm,
        ax=axes[0][1],
        hue_order=data.prop.unique(),
        style_order=data.prop.unique(),
    )

    ax.get_legend().remove()

    # sort or reorder the labels and handles
    ax.set_xscale("log")
    ax.set_xticks([0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
    ax.set_xticklabels([0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
    ax.set_xlabel("Evidence from Spurious-only Examples\n ($s$-only example rate)")
    ax.set_ylabel("$Both$ Error")

    ################# lineplot (a)
    _data = data
    _mean = _data.groupby(["rate", "prop"]).mean().reset_index()

    Y = "neither-error"
    ax = sns.lineplot(
        x="rate",
        y=Y,
        hue=MEASURE,
        dashes=False,
        markers=filled_markers,
        style="prop",
        linewidth=1.7,
        data=_mean,
        alpha=1,
        palette=cmap,
        hue_norm=norm,
        ax=axes[1][1],
        hue_order=data.prop.unique(),
        style_order=data.prop.unique(),
    )

    ax.get_legend().remove()

    # sort or reorder the labels and handles
    ax.set_xscale("log")
    ax.set_xticks([0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
    ax.set_xticklabels([0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
    ax.set_xlabel("Evidence from Spurious-only Examples\n ($s$-only example rate)")
    ax.set_ylabel("$Neither$ Error")

    ################# lineplot (a)
    _data = data
    _mean = _data.groupby(["rate", "prop"]).mean().reset_index()

    Y = "weak-error"
    ax = sns.lineplot(
        x="rate",
        y=Y,
        hue=MEASURE,
        dashes=False,
        markers=filled_markers,
        style="prop",
        linewidth=1.7,
        data=_mean,
        alpha=1,
        palette=cmap,
        hue_norm=norm,
        ax=axes[2][1],
        hue_order=data.prop.unique(),
        style_order=data.prop.unique(),
    )

    ax.get_legend().remove()

    # sort or reorder the labels and handles
    ax.set_xscale("log")
    ax.set_xticks([0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
    ax.set_xticklabels([0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
    ax.set_xlabel("Evidence from Spurious-only Examples\n ($s$-only example rate)")
    ax.set_ylabel("Spurious ($s$-only) Error")

    ################# lineplot (a)
    _data = data
    _mean = _data.groupby(["rate", "prop"]).mean().reset_index()

    Y = "test_f_score"
    ax = sns.lineplot(
        x="rate",
        y=Y,
        hue=MEASURE,
        dashes=False,
        markers=filled_markers,
        style="prop",
        linewidth=1.7,
        data=_mean,
        alpha=1,
        palette=cmap,
        hue_norm=norm,
        ax=axes[3][1],
        hue_order=data.prop.unique(),
        style_order=data.prop.unique(),
    )

    ax.get_legend().remove()

    # sort or reorder the labels and handles
    ax.set_xscale("log")
    ax.set_xticks([0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
    ax.set_xticklabels([0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5])
    ax.set_xlabel("Evidence from Spurious-only Examples\n ($s$-only example rate)")
    ax.set_ylabel("Test F-Score")

    model_name = MODEL.replace("-", "_").split("_")[0]
    plt.savefig(f"figures/all-{model_name}.pdf")  # .png", dpi=600)
    plt.close()


def plot_fulls(probing, results):
    Y = "weak-error"
    Y_label = "Use of Spurious Feature"

    _mean = probing.groupby(["prop"]).mean().reset_index()
    _MIN, _MAX = _mean["weak%strong-mdl"].min(), _mean["weak%strong-mdl"].max()
    plot_full(
        "weak%strong-mdl", "mdl", Y, "t5-base", Y_label, _MIN, _MAX, results, probing
    )  # test_f_score
    plot_full(
        "weak%strong-mdl",
        "mdl",
        Y,
        "bert-base-uncased",
        Y_label,
        _MIN,
        _MAX,
        results,
        probing,
    )
    plot_full(
        "weak%strong-mdl", "mdl", Y, "lstm-glove", Y_label, _MIN, _MAX, results, probing
    )
    plot_full(
        "weak%strong-mdl",
        "mdl",
        Y,
        "roberta-base",
        Y_label,
        _MIN,
        _MAX,
        results,
        probing,
    )
    plot_full(
        "weak%strong-mdl", "mdl", Y, "gpt2", Y_label, _MIN, _MAX, results, probing
    )


def main():
    results = pd.read_table("files/results.tsv")
    probing = pd.read_table("files/probing.tsv")
    results = results[results.accuracy_weak > 0.9]
    results = results[results.accuracy_strong > 0.9]

    probing = probing[probing.accuracy_weak > 0.9]
    probing = probing[probing.accuracy_strong > 0.9]

    # plot 1-offs
    plot_singles(probing, results)

    # plot full data
    plot_fulls(probing, results)
