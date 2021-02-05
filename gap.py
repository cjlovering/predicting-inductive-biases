import json
import os
import random

import inflect

import numpy as np
import pandas as pd
import plac
import pyinflect
import spacy
from sklearn.model_selection import train_test_split

import properties

random.seed(0)
np.random.seed(0)

with open("lexicon.json", "r") as f:
    data = json.load(f)
model = "en_core_web_lg"
nlp = spacy.load(model)
p = inflect.engine()


@plac.opt(
    "template", "prop to use", choices=["base", "hard"],
)
@plac.opt(
    "weak",
    "additional weak feature to use",
    choices=["none", "length", "lexical", "plural", "tense"],
)
@plac.opt(
    "splitcount", "number of examples in train / test",
)
def main(
    template="base",
    weak="length",
    splitcount=1000,
    rates=[0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5],
):
    """Produces filler-gap examples with `prop` as the counter example.

    This will generate the files needed for probing and finetuning.

    TODO: Generate an all option. We have to figure out how to handle cases
    with both positive and negative counter examples.

    NOTE: The val data is distributed as the trained data (with the supplied `rate` of
    counter examples).

    NOTE: The test data isn't balanced but includes many examples of the prop
    types. We will partition the test set so balancing is not very important.

    NOTE: Currently, the val and test data overlap. If we turn off early stopping
    which may be a good idea for the auc anyway, then we have no issue.

    NOTE: Set a column `label` to be used per class.
    """
    assert not (
        template == "base" and weak == "none"
    ), "We don't have a weak property here."

    # 2.5 as there many be some duplicates and we want section_size for both train and test.
    section_size = splitcount
    count = round(2.5 * section_size)
    if not os.path.exists("./properties"):
        os.mkdir("./properties")
    if not os.path.exists(f"./properties/gap-{template}-{weak}/"):
        os.mkdir(f"./properties/gap-{template}-{weak}/")

    both_templates = [
        ("S_wh_gap", "both", "yes", S_wh_gap),
        ("S_that_no_gap", "both", "yes", S_that_no_gap),
    ]
    neither_templates = [
        ("S_wh_no_gap", "neither", "no", S_wh_no_gap),
        ("S_that_gap", "neither", "no", S_that_gap),
    ]
    counter_templates = []
    if weak != "none":
        counter_templates.extend(
            [
                ("S_wh_no_gap", "weak", "no", S_wh_no_gap),
                ("S_that_gap", "weak", "no", S_that_gap),
            ]
        )
    if template == "hard":
        counter_templates.append(("S_island", "weak", "no", S_island),)
    if weak == "length":
        min_both_N = max_both_N = min_weak_N = max_weak_N = 3
        min_neither_N = max_neither_N = 2
        include_continuation_both = include_continuation_weak = True
        include_continuation_neither = False
    else:
        min_both_N = min_neither_N = min_weak_N = 1
        max_both_N = max_neither_N = max_weak_N = 3
        include_continuation_neither = (
            include_continuation_both
        ) = include_continuation_weak = True

    if weak == "lexical":
        force_subject_both = force_subject_weak = True
        force_subject_neither = False
    else:
        force_subject_both = force_subject_weak = force_subject_neither = False

    if weak == "tense":
        force_past_both = force_past_weak = True
        force_past_neither = False
    else:
        force_past_both = force_past_weak = force_past_neither = False

    if weak == "plural":
        force_plural_both = True
        force_plural_weak = True
        force_plural_neither = False
    else:
        force_plural_both = force_plural_weak = force_plural_neither = False

    output = []
    for name, section, acceptable, S_template in both_templates:
        for _ in range(count):
            sentence = S_template(
                min_both_N,
                max_both_N,
                include_continuation_both,
                force_past_both,
                force_subject_both,
                force_plural_both,
            )
            output.append(
                {
                    "sentence": sentence,
                    "section": section,
                    "acceptable": acceptable,
                    "template": name,
                }
            )
    for name, section, acceptable, S_template in neither_templates:
        for _ in range(count):
            sentence = S_template(
                min_neither_N,
                max_neither_N,
                include_continuation_neither,
                force_past_neither,
                force_subject_neither,
                force_plural_neither,
            )
            output.append(
                {
                    "sentence": sentence,
                    "section": section,
                    "acceptable": acceptable,
                    "template": name,
                }
            )
    counter_output = []
    for name, section, acceptable, S_template in counter_templates:
        for _ in range(count):
            sentence = S_template(
                min_weak_N,
                max_weak_N,
                include_continuation_weak,
                force_past_weak,
                force_subject_weak,
                force_plural_weak,
            )
            counter_output.append(
                {
                    "sentence": sentence,
                    "section": section,
                    "acceptable": acceptable,
                    "template": name,
                }
            )
    counter_df = pd.DataFrame(counter_output)
    counter_df = counter_df.sort_values(["acceptable", "section", "template"])
    counter_df = counter_df.drop_duplicates("sentence")
    counter_df["label"] = (counter_df.acceptable == "yes").astype(int)
    train_counterexample, test_counterexample = train_test_split(
        counter_df, test_size=0.5
    )

    df = pd.DataFrame(output)
    df = df.sort_values(["acceptable", "section", "template"])
    df = df.drop_duplicates("sentence")
    # NOTE: This label is the acceptable label used for finetuning
    # This label will be over-written later when the probing splits are generated.
    df["label"] = (df.acceptable == "yes").astype(int)
    train_base, test_base = train_test_split(df, test_size=0.5)
    counter_section = "weak"
    properties.generate_property_data(
        f"gap-{template}-{weak}",
        counter_section,
        train_base,
        test_base,
        train_counterexample,
        test_counterexample,
        section_size,
        rates,
    )


def S(
    words,
    include_object,
    include_continuation,
    force_past,
    force_subject,
    force_plural,
    splice_level=-1,
):
    out = get_complement("prefix_verb", force_subject, force_plural)

    for i, w in enumerate(words):
        out.append(w)
        if force_past:
            out += get_complement("verb_past")
        else:
            out += get_complement("verb")
        if splice_level == i:
            out.append(random.choice(data["object"]))

    if include_object:
        out.append(random.choice(data["object"]))

    if include_continuation:
        out.append(random.choice(data["continuation"]))

    return stringify(out)


def get_complement(verb_section, force_subject=False, force_plural=False):
    out = []
    if force_subject:
        out.append("I")
        # She knows. We know. I know.
        subj_is_plural = True
    else:
        if force_plural:
            subj_temp = random.choice(
                [
                    ["<PLURAL-NOUN1>"],
                    ["<PLURAL-ARTICLE>", "<PLURAL-NOUN2>"],
                    ["<PLURAL-NOUN2>"],
                ]
            )
        else:
            subj_temp = random.choice(data["subj_temp"])

        subj_is_plural = False
        for part in subj_temp:
            subj_is_plural = "PLURAL" in part
            out.append(random.choice(data[part]))

    verb = random.choice(data[verb_section])
    if subj_is_plural:
        out.append(p.plural_verb(verb))
    else:
        out.append(verb)

    return out


def stringify(sent):
    sent = " ".join(sent).replace(" ,", ",")
    sent = sent[0].upper() + sent[1:]
    return sent


def S_wh_gap(
    min_N, max_N, include_continuation, force_past, force_subject, force_plural
):
    N = random.randint(min_N, max_N)
    words = ["that"] * (N - 1) + ["who"]
    random.shuffle(words)
    return S(
        words,
        include_object=False,
        include_continuation=include_continuation,
        force_past=force_past,
        force_subject=force_subject,
        force_plural=force_plural,
    )


def S_that_no_gap(
    min_N, max_N, include_continuation, force_past, force_subject, force_plural
):
    N = random.randint(min_N, max_N)
    words = ["that"] * (N)
    random.shuffle(words)
    return S(
        words,
        include_object=True,
        include_continuation=include_continuation,
        force_past=force_past,
        force_subject=force_subject,
        force_plural=force_plural,
    )


def S_wh_no_gap(
    min_N, max_N, include_continuation, force_past, force_subject, force_plural
):
    N = random.randint(min_N, max_N)
    words = ["that"] * (N - 1) + ["who"]
    random.shuffle(words)
    return S(
        words,
        include_object=True,
        include_continuation=include_continuation,
        force_past=force_past,
        force_subject=force_subject,
        force_plural=force_plural,
    )


def S_that_gap(
    min_N, max_N, include_continuation, force_past, force_subject, force_plural
):
    N = random.randint(min_N, max_N)
    words = ["that"] * (N)
    random.shuffle(words)
    return S(
        words,
        include_object=False,
        include_continuation=include_continuation,
        force_past=force_past,
        force_subject=force_subject,
        force_plural=force_plural,
    )


def S_island(
    min_N, max_N, include_continuation, force_past, force_subject, force_plural
):
    N = random.randint(max(min_N, 2), max_N)

    if N == 2:
        splice_level = 0
        words = ["who", "that"]
    elif N == 3:
        if random.random() < 0.67:
            splice_level = 1
            words = random.choice([["who", "that", "that"], ["that", "who", "that"]])
        else:
            splice_level = 0
            words = ["who", "that", "that"]
    else:
        print(N)
        assert False

    return S(
        words,
        include_object=False,
        include_continuation=include_continuation,
        force_past=force_past,
        splice_level=splice_level,
        force_subject=force_subject,
        force_plural=force_plural,
    )


if __name__ == "__main__":
    plac.call(main)
