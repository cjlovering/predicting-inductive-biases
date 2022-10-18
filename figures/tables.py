import pandas as pd
from typing import List


def compute_correlations(results: pd.DataFrame, target_metric: str) -> pd.DataFrame:
    """Formats the data for 1) scatterplot plotting, 2) correlations."""

    # feature use
    df = results.copy()
    # df = df[(df.prop != "gap-hard-plural") & (df.prop != "gap-base-plural")]

    df["strong-auc"] = df.val_loss_auc_strong.astype("float")
    df["weak-auc"] = df.val_loss_auc_weak.astype("float")

    # Reading the docs, it seems that spearman does NOT expect values to be normalized.
    NORM = lambda x: x
    df["weak%strong-mdl"] = NORM(df["total_mdl_weak"] / df["total_mdl_strong"])
    df["weak%strong-auc"] = NORM(df["weak-auc"] / df["strong-auc"])
    df["weak-strong-mdl"] = NORM(df["total_mdl_weak"] - df["total_mdl_strong"])
    df["weak-strong-auc"] = NORM(df["weak-auc"] - df["strong-auc"])
    df["strong-mdl"] = NORM(df["total_mdl_strong"].astype("float"))
    df["strong-auc"] = NORM(df.val_loss_auc_strong.astype("float"))
    df["weak-mdl"] = NORM(df["total_mdl_weak"].astype("float"))
    df["weak-auc"] = NORM(df.val_loss_auc_weak.astype("float"))

    if target_metric == "evidence_required":
        EVIDENCE_REQUIRED_THRESHOLD = 0.99
        correlations = df[df["test_f_score"] > EVIDENCE_REQUIRED_THRESHOLD]
    else:
        correlations = df  # IF USING FSCORE

    def fun(rate):
        return min(rate)

    correlations = (
        correlations.groupby(
            [
                "model",
                "prop",
                "rate",
            ]
        )
        .mean()
        .reset_index()
        .groupby(["model", "prop"])
        .apply(
            lambda x: pd.Series(
                {
                    "first": fun(x.rate),
                    "test_f_score": x["test_f_score"].mean(),
                    "f_area": x["test_f_score"].sum(),
                    "f_error_area": (1 - x["test_f_score"]).sum(),
                    "weak%strong-mdl": x["weak%strong-mdl"].iloc[0],
                    "weak%strong-acc": x["weak%strong-acc"].iloc[0],
                    "weak%strong-auc": x["weak%strong-auc"].iloc[0],
                    "weak-strong-mdl": x["weak-strong-mdl"].iloc[0],
                    "weak-strong-auc": x["weak-strong-auc"].iloc[0],
                    "weak-strong-acc": x["weak-strong-acc"].iloc[0],
                    "accuracy_weak": x["accuracy_weak"].iloc[0],
                    "accuracy_strong": x["accuracy_strong"].iloc[0],
                    "weak-mdl": x["weak-mdl"].iloc[0],
                    "weak-auc": x["weak-auc"].iloc[0],
                    "strong-mdl": x["strong-mdl"].iloc[0],
                    "strong-auc": x["strong-auc"].iloc[0],
                }
            )
        )
        .reset_index()
    )

    import scipy

    scatterdata = correlations.copy()

    CORR = scipy.stats.spearmanr
    correlations = correlations[correlations.prop.apply(lambda x: "toy" not in x)]
    correlations.to_csv("files/correlations.tsv", sep="\t", index=False)

    out = []
    raw = []
    for m in correlations.model.unique():
        data = correlations[(correlations.model == m)]
        #     target = data["first"]
        target = data["test_f_score"]
        for measure in [
            "weak%strong-acc",
            "weak%strong-mdl",
            "weak%strong-auc",
            "weak-strong-acc",
            "weak-strong-mdl",
            "weak-strong-auc",
            "weak-mdl",
            "weak-auc",
            "strong-mdl",
            "strong-auc",
            "accuracy_weak",
            "accuracy_strong",
        ]:
            #         if measure == "weak%strong-mdl":
            #             vals = data[measure] > 1
            #         if measure == "weak-strong-mdl":
            #             vals = data[measure] > 0
            #         print(list(data))
            corr, p_value = CORR(data[measure], target)
            #         corr, p_value = CORR(vals, target)
            out.append(
                {
                    "model": m,
                    "measure": measure,
                    "corr": corr,
                    "p_value": p_value,
                }
            )

    data = pd.DataFrame(out)
    # data = data[data.measure.str.contains("mdl")]

    def summarize(x):
        if x["p_value"] < 0.05:
            return f"{x['corr']}*"
        else:
            return f"{x['corr']}"

    #     f"{x['corr']}, {x['p_value']}"

    data["corr"] = data["corr"].apply(lambda x: round(x, 2))
    data["p_value"] = data.p_value.apply(lambda x: round(x, 2))
    data["out"] = data.apply(summarize, axis=1)

    tbl = data.pivot(index="model", columns="measure", values="out").reset_index()[
        [
            "model",
            "weak%strong-mdl",
            "weak-strong-mdl",
            "strong-auc",
            "weak-auc",
            "strong-mdl",
            "weak-mdl",
            "weak%strong-auc",
            "weak-strong-auc",
            "weak%strong-acc",
            "weak-strong-acc",
            "accuracy_weak",
            "accuracy_strong",
        ]
    ]
    return scatterdata, tbl


def format_table(
    title: str,
    correlations: pd.DataFrame,
    column_order: List[str],
) -> None:
    r"""Prints out a table. You'll have to remove the insignificant stats yourself.
    You'll have to move it into the proper \begin{table} yourself.

    Input:
    \begin{tabular}{llllll}
    \toprule
    measure &              model &   strong-auc &     weak-auc & weak\%strong-auc & weak-strong-auc \\
    \midrule
    0 &  bert-base-uncased &  -0.62, 0.02 &   0.56, 0.04 &       0.73, 0.0 &      0.66, 0.01 \\
    1 &               gpt2 &  -0.34, 0.17 &   0.52, 0.03 &       0.64, 0.0 &       0.67, 0.0 \\
    2 &         lstm-glove &  -0.51, 0.11 &  -0.51, 0.11 &     -0.18, 0.59 &     -0.11, 0.75 \\
    3 &             rebert &  -0.09, 0.87 &   0.26, 0.62 &      0.03, 0.96 &      0.09, 0.87 \\
    4 &       roberta-base &  -0.12, 0.63 &    0.67, 0.0 &       0.75, 0.0 &       0.64, 0.0 \\
    5 &            t5-base &   -0.79, 0.0 &   0.56, 0.04 &       0.92, 0.0 &       0.97, 0.0 \\
    \bottomrule
    \end{tabular}

    """
    readability_swaps = {
        "bert-base-uncased": "BERT",
        "t5-base": "T5",
        "roberta-base": "RoBERTa",
        "gpt2": "GPT2",
        "lstm-glove": "GloVe",
        "llllll": "lllll",
        "measure &": "",
        "model": "",
        **{f"\n{i} &": "\n" for i in range(10)},
    }

    tbl = correlations[column_order]
    tbl = tbl.set_index("model")
    # tbl = tbl.reindex(["BERT", "RoBERTa", "T5", "GPT2", "GloVe",])
    tbl_text = tbl.to_latex()

    # The double slash creates a slash, the triple { creates a bracket + lets us insert the title.
    output = f"""{tbl_text}\\caption{{{title}}}\n"""
    for k, v in readability_swaps.items():
        output = output.replace(k, v)
    print(output)


def main():
    # uncontrolled
    results = pd.read_table("files/results.tsv")
    _, fscore_tbl_uncontrolled = compute_correlations(results, "fscore")

    # controlled
    results = pd.read_table("files/results.tsv")
    PROBING_SOLVED_THRESHOLD = 0.9
    results = results[results.accuracy_weak > PROBING_SOLVED_THRESHOLD]
    results = results[results.accuracy_strong > PROBING_SOLVED_THRESHOLD]
    scatterdata, fscore_tbl = compute_correlations(results, "fscore")
    _, evidencerequired_tbl = compute_correlations(results, "evidence_required")

    # save intermediate scatterplot data
    scatterdata.to_csv("files/scatterdata.tsv", sep="\t", index=False)

    # print out tables
    format_table(
        "uncontrolled-mdl",
        fscore_tbl_uncontrolled,
        ["model", "strong-mdl", "weak-mdl", "weak%strong-mdl", "weak-strong-mdl"],
    )
    format_table(
        "average f score vs mdl",
        fscore_tbl,
        ["model", "strong-mdl", "weak-mdl", "weak%strong-mdl", "weak-strong-mdl"],
    )
    format_table(
        "evidence required vs mdl",
        evidencerequired_tbl,
        ["model", "strong-mdl", "weak-mdl", "weak%strong-mdl", "weak-strong-mdl"],
    )
    format_table(
        "average f score vs auc",
        fscore_tbl,
        ["model", "strong-auc", "weak-auc", "weak%strong-auc", "weak-strong-auc"],
    )
