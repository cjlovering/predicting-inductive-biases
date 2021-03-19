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
import tqdm


def collect(task) -> pd.DataFrame:
    """Loads results from disk. """
    output = []
    # NOTE: Asssumes all relevant files are in the path.
    result_path = f"../results/stats/*{task}*"
    for result_file_name in glob.glob(result_path):
        x = pd.read_table(result_file_name)
        if "probe" in task:
            if "strong_direct" in result_file_name:
                x["probe"] = "strong_direct"
            elif "strong" in result_file_name:
                x["probe"] = "strong"
            elif "weak" in result_file_name:
                x["probe"] = "weak"
        output.append(x)
    return pd.concat(output)


def collect_probing_accuracy() -> pd.DataFrame:
    """The accuracy in the probing stats is the accuracy on the full test set, not of the probing class.
    We can recreate the accuracy from the raw outputs.
    """
    import glob

    score = lambda x: x["correct"].mean()

    result = []
    for f in tqdm.tqdm(
        glob.glob("../results/raw/*.tsv"), desc="Computing Probing Test Accuracy"
    ):
        # skip pilot studies
        if "block" in f:
            continue
        if "msgs" in f:
            continue

        # skip finetune, etc
        if "probing" not in f:
            continue

        # Load table and figure out the model.
        df = pd.read_table(f)
        for m in [
            "t5-base",
            "roberta-base",
            "bert-base-uncased",
            "lstm-toy",
            "rebert",
            "gpt2",
            "lstm-glove",
            "elmo",
        ]:
            if m in f:
                model = m
                break

        # Determine which probe we're looking at.
        if "strong" in f:
            probe = "strong"
        else:
            probe = "weak"

        # Determine the seed
        seed = int(f.split("_")[-1].replace(".tsv", ""))
        df["correct"] = df["pred"] == df["label"]

        # Splice out partitions.
        target_data = df[(df.section == "both") | (df.section == "weak")]
        spurious_data = df[(df.section == "neither") | (df.section == "weak")]

        if probe == "strong":
            accuracy = score(target_data)
        else:
            # Set the weak data labels to be 1 for weak, as its weak vs neither.
            spurious_data.loc[(spurious_data.section == "weak"), "label"] = 1
            spurious_data["correct"] = spurious_data["pred"] == spurious_data["label"]
            accuracy = score(spurious_data)

        # get the prop name
        title = "_".join(f.split("_")[:3])
        if "npi" in f:
            title = "_".join(f.split("_")[:2])
        prop = (
            title.replace("../results/raw/", "")
            .replace("_probing_weak", "")
            .replace("_probing_strong", "")
            .replace("_probing", "")
        )
        if "npi" in f and prop not in [
            "npi_length",
            "npi_lexical",
            "npi_plural",
            "npi_tense",
        ]:
            assert False, prop

        # save the result
        result.append(
            {
                "prop": prop,
                "model": model,
                "accuracy": accuracy,
                "probe": probe,
                "seed": seed,
            }
        )
    df = pd.DataFrame(result)

    # format it.
    probing_accuracy_1 = (
        df.groupby(["prop", "model", "probe", "seed"])
        .mean()[["accuracy"]]
        .reset_index()
    )
    probing_accuracy_2 = pd.merge(
        probing_accuracy_1[probing_accuracy_1.probe == "weak"],
        probing_accuracy_1[probing_accuracy_1.probe == "strong"],
        left_on=["model", "prop", "seed"],
        right_on=["model", "prop", "seed"],
        suffixes=["_weak", "_strong"],
    )
    probing_accuracy_3 = probing_accuracy_2[
        ["model", "prop", "accuracy_weak", "accuracy_strong", "seed"]
    ]
    return probing_accuracy_3


def organize_data():
    probing = collect("probing")
    probing = (
        probing.groupby(["model", "prop", "probe"])
        .mean()
        .reset_index()[["model", "prop", "probe", "val_loss_auc", "total_mdl"]]
    )
    finetune = collect("finetune")

    probing_1 = probing[["model", "prop", "probe", "val_loss_auc", "total_mdl"]]
    probing_2 = pd.merge(
        probing_1[probing_1.probe == "weak"],
        probing_1[probing_1.probe == "strong"],
        left_on=["model", "prop"],
        right_on=["model", "prop"],
        suffixes=["_weak", "_strong"],
    )
    probing_3 = probing_2[
        [
            "model",
            "prop",
            "val_loss_auc_weak",
            "val_loss_auc_strong",
            "total_mdl_weak",
            "total_mdl_strong",
        ]
    ]

    # Load and mean probing accuracy (over random seeds) as the df of main results
    # random seeds don't really match up--its not the same run.
    probing_accuracy = (
        pd.read_table("files/probing_accuracy.tsv").groupby(["model", "prop"]).mean()
    )  # mean out seed
    probing_accuracy = probing_accuracy.reset_index()[
        ["model", "prop", "accuracy_weak", "accuracy_strong"]
    ]
    probing_4 = pd.merge(
        probing_3,
        probing_accuracy,
        left_on=["model", "prop"],
        right_on=["model", "prop"],
    )

    df = pd.merge(
        probing_4, finetune, left_on=["model", "prop"], right_on=["model", "prop"]
    )
    df["strong-auc"] = df.val_loss_auc_strong.astype("float").div(1000)
    df["weak-auc"] = df.val_loss_auc_weak.astype("float").div(1000)
    df["weak-strong-auc"] = df["weak-auc"] - df["strong-auc"]
    df["strong-weak-auc"] = df["strong-auc"] - df["weak-auc"]
    df["weak%strong-auc"] = df["weak-auc"] / df["strong-auc"]

    df["strong-mdl"] = df["total_mdl_strong"].astype("float").div(1000)
    df["weak-mdl"] = df["total_mdl_weak"].astype("float").div(1000)
    df["weak-strong-mdl"] = df["total_mdl_weak"] - df["total_mdl_strong"]
    df["weak%strong-mdl"] = df["weak-mdl"] / df["strong-mdl"]

    # df["strong-mdl"] = df["total_mdl_strong"].astype("float").div(1000)
    # df["weak-mdl"] = df["total_mdl_weak"].astype("float").div(1000)
    df["weak-strong-acc"] = df["accuracy_weak"] - df["accuracy_strong"]
    df["weak%strong-acc"] = df["accuracy_weak"] / df["accuracy_strong"]

    results = df
    results = results[results.prop != "toy_4"]

    results.to_csv("files/results.tsv", index=False, sep="\t")
    results.groupby(["model", "prop"]).mean().reset_index().to_csv(
        "files/results_avg.tsv", sep="\t", index=False
    )

    probing = collect("probing")
    probing = probing[results.prop != "toy_4"]
    probing.to_csv("files/raw.tsv", sep="\t", index=False)

    probing_1 = probing[["seed", "model", "prop", "probe", "val_loss_auc", "total_mdl"]]
    probing_2 = pd.merge(
        probing_1[probing_1.probe == "weak"],
        probing_1[probing_1.probe == "strong"],
        left_on=["seed", "model", "prop"],
        right_on=["seed", "model", "prop"],
        suffixes=["_weak", "_strong"],
    )
    probing_full = probing_2[
        [
            "model",
            "prop",
            "seed",
            "val_loss_auc_weak",
            "val_loss_auc_strong",
            "total_mdl_weak",
            "total_mdl_strong",
        ]
    ]
    probing_accuracy = pd.read_table("files/probing_accuracy.tsv")
    probing_full = pd.merge(
        probing_full,
        probing_accuracy,
        left_on=["model", "prop", "seed"],
        right_on=["model", "prop", "seed"],
    )

    probing_full["strong-auc"] = probing_full.val_loss_auc_strong.astype("float").div(
        1000
    )
    probing_full["weak-auc"] = probing_full.val_loss_auc_weak.astype("float").div(1000)
    probing_full["weak-strong-auc"] = (
        probing_full["weak-auc"] - probing_full["strong-auc"]
    )
    probing_full["strong-weak-auc"] = (
        probing_full["strong-auc"] - probing_full["weak-auc"]
    )
    probing_full["weak%strong-auc"] = (
        probing_full["weak-auc"] / probing_full["strong-auc"]
    )

    probing_full["strong-mdl"] = (
        probing_full["total_mdl_strong"].astype("float").div(1000)
    )
    probing_full["weak-mdl"] = probing_full["total_mdl_weak"].astype("float").div(1000)
    probing_full["weak-strong-mdl"] = (
        probing_full["total_mdl_weak"] - probing_full["total_mdl_strong"]
    )
    probing_full["weak%strong-mdl"] = (
        probing_full["weak-mdl"] / probing_full["strong-mdl"]
    )

    probing_full["weak-strong-acc"] = (
        probing_full["accuracy_weak"] - probing_full["accuracy_strong"]
    )
    probing_full["weak%strong-acc"] = (
        probing_full["accuracy_weak"] / probing_full["accuracy_strong"]
    )

    probing_full.to_csv("files/probing.tsv", sep="\t", index=False)
    return probing


def main():

    # output intermediate files.
    probing_accuracy = collect_probing_accuracy()
    probing_accuracy.to_csv("files/probing_accuracy.tsv", sep="\t", index=False)

    organize_data()
