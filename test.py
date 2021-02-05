import glob

import numpy as np
import pandas as pd
import pytest

import job


def find_probing_files():
    return list(glob.glob("properties/*/probing*.tsv"))


def find_finetune_files():
    files = list(glob.glob("properties/*/finetune*.tsv"))
    files_rates = [(f, float(f.split("_")[-2])) for f in files]
    return files_rates


def find_test_files():
    return list(glob.glob("properties/*/test.tsv"))


@pytest.mark.parametrize("path", find_probing_files())
def test_data_probe(path):
    """Checks that the data is an even split over the `label` column. """
    df = pd.read_table(path)
    assert {"sentence", "section", "label"}.issubset(
        set(list(df))
    ), "We require these columns."
    assert set(df.section.unique()).issubset(
        {"neither", "both", "weak", "strong"}
    ), "Use this terminology."
    label_counts = df.groupby("label").count()["sentence"]
    assert (
        len(label_counts) >= 2
    ), f"There must be at least two labels; there are only {len(label_counts)}."
    assert (
        len(label_counts.unique()) == 1
    ), f"We expect the number of each label to be equal."


@pytest.mark.parametrize("path,rate", find_finetune_files())
def test_data_finetune(path, rate):
    """Checks that the data is an even split over the `label` column. """
    df = pd.read_table(path)
    assert {"sentence", "section", "label"}.issubset(
        set(list(df))
    ), "We require these columns."
    assert set(df.section.unique()).issubset(
        {"neither", "both", "weak", "strong"}
    ), "Use this terminology."
    df["base"] = (df.section == "both") | (df.section == "neither")
    # This test may fail if your datasets are small < 100 per section.
    assert np.isclose(df["base"].mean(), 1 - rate, atol=0.005)


@pytest.mark.parametrize("path", find_test_files())
def test_data_test(path):
    """Checks that the data is an even split over the `label` column. """
    df = pd.read_table(path)
    assert {"sentence", "section", "label"}.issubset(
        set(list(df))
    ), "We require these columns."
    assert set(df.section.unique()).issubset(
        {"neither", "both", "weak", "strong"}
    ), "Use this terminology."


@pytest.mark.parametrize(
    "prop, rate, probe, task, model, expected",
    [
        ["toy_1", 0.5, "strong", "probing", "lstm-toy", False],
        ["toy_1", 0.5, "strong", "probing", "lstm-glove", True],
        ["sva", 0.5, "strong", "probing", "lstm-toy", True],
        ["sva", 0.5, "strong", "probing", "lstm-glove", False],
    ],
)
def test_data_test(prop, rate, probe, task, model, expected):
    """Checks that the data is an even split over the `label` column. """
    assert job.filter_option_out(prop, rate, probe, task, model, 1) == expected
