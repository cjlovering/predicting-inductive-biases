import math

import pandas as pd


def generate_property_data(
    prop,
    counter_section,
    train_base,
    test_base,
    train_counterexample,
    test_counterexample,
    section_size,
    rates,
    test_section_size: int = None,
):
    """See `gap.py` for an example use case.

    Parameters
    ----------
    ``prop``: string
        The name of the prop
    ``counter_section``: str
        The section of counterexample data. It should be either `strong` or `weak`.
        TODO: Refactor to allow both strong and weak counter example.
    ``train_base``: pd.DataFrame
        both/neither training data
    ``test_base``: pd.DataFrame
        strong/weak test data
    ``train_counterexample``: pd.DataFrame
        Counterexample training data
    ``test_counterexample``: pd.DataFrame
        Counterexample test data
    ``section_size``: int
        The number of examples from each split.
    ``rates``: List[float]
        The rates to be generated
    ``test_section_size``: Optional(int), default = None
        The number of examples from each split (for testing).
        If None, set to the section size.

    NOTES
    -----

    1. data format is `.tsv`

        The data is a `.tsv` format: with a `sentence`, `section` and `label` column.

        The `sentence` is the sentence, the `section` is one of (neither, both, weak, strong),
        and the `label` is 0 or 1. This allows us to use the same pipeline for the probing and finetuning.

        ```
        # This is an example. Any additional columns are no problem and will be tracked/kept together,
        # esp. with the test data for analysis.

        sentence	section	acceptable	template	parenthetical_count	clause_count	label
        Guests hoped who guests determined him last week	neither	no	S_wh_no_gap	0	1	0
        Teachers believe who you held before the trial	both	yes	S_wh_gap	0	1	1
        You think that guests determined that visitors recommended someone over the summer	both	yes	S_that_no_gap	0	2	1
        Professors believe that professors loved over the summer	neither	no	S_that_gap	0	1	0
        ```

    2. data files are saved as
        ```
        # finetune
        path = f"{task}_{rate}"
        # probing
        path = f"{task}_{feature}"
        "./properties/{prop}/{path}_train.tsv"
        "./properties/{prop}/{path}_val.tsv"
        "./properties/{prop}/test.tsv"
        ```
    """
    if test_section_size is None:
        test_section_size = section_size
    # Weak probing.
    if counter_section == "weak":
        # Neither vs Weak
        target_section = "weak"
        other_section = "neither"

        weak_probing_train = probing_split(
            train_base,
            train_counterexample,
            section_size,
            target_section,
            other_section,
        )
        weak_probing_test = probing_split(
            test_base,
            test_counterexample,
            test_section_size,
            target_section,
            other_section,
        )

        weak_probing_train.to_csv(
            f"./properties/{prop}/probing_weak_train.tsv", index=False, sep="\t"
        )
        weak_probing_test.to_csv(
            f"./properties/{prop}/probing_weak_val.tsv", index=False, sep="\t"
        )
    else:
        # Both vs Strong
        target_section = "both"
        other_section = "strong"

        weak_probing_train = probing_split(
            train_base,
            train_counterexample,
            section_size,
            target_section,
            other_section,
        )
        weak_probing_test = probing_split(
            test_base,
            test_counterexample,
            test_section_size,
            target_section,
            other_section,
        )

        weak_probing_train.to_csv(
            f"./properties/{prop}/probing_weak_train.tsv", index=False, sep="\t"
        )
        weak_probing_test.to_csv(
            f"./properties/{prop}/probing_weak_val.tsv", index=False, sep="\t"
        )

    # Strong probing.
    if counter_section == "strong":
        # Neither vs Strong
        target_section = "strong"
        other_section = "neither"

        strong_probing_train = probing_split(
            train_base,
            train_counterexample,
            section_size,
            target_section,
            other_section,
        )
        strong_probing_test = probing_split(
            test_base,
            test_counterexample,
            test_section_size,
            target_section,
            other_section,
        )

        strong_probing_train.to_csv(
            f"./properties/{prop}/probing_strong_train.tsv", index=False, sep="\t"
        )
        strong_probing_test.to_csv(
            f"./properties/{prop}/probing_strong_val.tsv", index=False, sep="\t"
        )
    else:
        # Both vs Strong
        target_section = "both"
        other_section = "weak"

        strong_probing_train = probing_split(
            train_base,
            train_counterexample,
            section_size,
            target_section,
            other_section,
        )
        strong_probing_test = probing_split(
            test_base,
            test_counterexample,
            test_section_size,
            target_section,
            other_section,
        )
        strong_probing_train.to_csv(
            f"./properties/{prop}/probing_strong_train.tsv", index=False, sep="\t"
        )
        strong_probing_test.to_csv(
            f"./properties/{prop}/probing_strong_val.tsv", index=False, sep="\t"
        )

    # set up fine-tuning.
    for rate in rates:
        finetune_train = finetune_split(
            train_base,
            train_counterexample,
            # We keep the probing and finetune set sizes the same, even though we#
            # could make the finetuning bigger.
            2 * section_size,
            rate,
        )
        finetune_val = finetune_split(
            test_base,
            test_counterexample,
            # We keep the probing and finetune set sizes the same, even though we#
            # could make the finetuning bigger.
            2 * test_section_size,
            rate,
        )
        finetune_train.to_csv(
            f"./properties/{prop}/finetune_{rate}_train.tsv",
            index=False,
            sep="\t",
        )
        finetune_val.to_csv(
            f"./properties/{prop}/finetune_{rate}_val.tsv",
            index=False,
            sep="\t",
        )

    # save test.
    test = pd.concat([test_base, test_counterexample])
    test.to_csv(f"./properties/{prop}/test.tsv", index=False, sep="\t")


def generate_property_data_strong_direct(
    prop,
    counter_section,
    train_base,
    test_base,
    train_counterexample,
    test_counterexample,
    section_size,
    rates,
    test_section_size: int = None,
):
    if test_section_size is None:
        test_section_size = section_size

    # Neither vs Strong
    target_section = "strong"
    other_section = "neither"

    strong_probing_train = probing_split(
        train_base,
        train_counterexample,
        section_size,
        target_section,
        other_section,
    )
    strong_probing_test = probing_split(
        test_base,
        test_counterexample,
        test_section_size,
        target_section,
        other_section,
    )

    strong_probing_train.to_csv(
        f"./properties/{prop}/probing_strong_direct_train.tsv", index=False, sep="\t"
    )
    strong_probing_test.to_csv(
        f"./properties/{prop}/probing_strong_direct_val.tsv", index=False, sep="\t"
    )


def probing_split(
    train_base,
    test_base,
    train_counterexample,
    test_counterexample,
    section_size,
    target_section,
    other_section,
):
    """Generate a split for probing target_section vs other_section where
    target_section is set as the positive section.
    """

    def filter_sample(df, section):
        return df[df.section == section].sample(section_size)

    train_data = pd.concat([train_base, train_counterexample])
    test_data = pd.concat([test_base, test_counterexample])
    train = pd.concat(
        [
            filter_sample(train_data, other_section),
            filter_sample(train_data, target_section),
        ]
    )
    test = pd.concat(
        [
            filter_sample(test_data, other_section),
            filter_sample(test_data, target_section),
        ]
    )
    train["label"] = (train.section == target_section).astype(int)
    test["label"] = (test.section == target_section).astype(int)
    train["label_str"] = train["label"].apply(lambda x: {0: "False", 1: "True"}[x])
    test["label_str"] = test["label"].apply(lambda x: {0: "False", 1: "True"}[x])
    return train, test


def probing_split(
    base,
    counterexample,
    section_size,
    target_section,
    other_section,
):
    """Generate a split for probing target_section vs other_section where
    target_section is set as the positive section.
    """

    def filter_sample(df, section):
        return df[df.section == section].sample(section_size)

    data = pd.concat([base, counterexample])
    data = pd.concat(
        [
            filter_sample(data, other_section),
            filter_sample(data, target_section),
        ]
    )
    data["label"] = (data.section == target_section).astype(int)
    return data


def get_config(config_path):
    section_to_configs = {"both": [], "neither": [], "weak": [], "strong": []}
    try:
        with open(config_path, "r") as f:
            df = pd.read_csv(f)
            df_as_dict = df.to_dict(orient="records")
            for config in df_as_dict:
                section = config["section"]
                section_to_configs[section].append(config)
    except OSError as e:
        print("No config file for this template.")
        raise (e)

    return section_to_configs


def finetune_split(base, counterexample, total_size, rate):
    size_base, size_target = (
        math.floor(total_size * (1.0 - rate)),
        math.ceil(total_size * rate),
    )
    finetune = pd.concat([base.sample(size_base), counterexample.sample(size_target)])
    return finetune
