# No authors that the security guards like have ever been famous
# The authors that the security guards like have not ever been famous
# *The authors that the security guards like have ever been famous
# *The authors that no security guards like have ever been famous
# *The authors that the security guards don’t like have ever been famous
# *The authors that the security guards like have ever not been famous

import json
import os
import random
import plac

import pandas as pd
from sklearn.model_selection import train_test_split

import properties

grammar = {
    "S-good": ["S1-good", "S1-good and S1-good"],
    "S-bad": ["S1-bad", "S1-good and S1-bad", "S1-bad and S1-good"],
    "S1-good": ["no NP-neg ever VB-intrans", "DT NP VB-intrans"],
    "S1-bad": ["DT NP ever VB-intrans", "DT NP-bad VB-intrans"],
    "NP": [
        "NP1",
        "NN1 who DT NP VB-trans",
        "NN1 who DT NP VB-trans",
        "NN1 who no NP-neg VB-trans",
        "NN1 who no NP-neg ever VB-trans",
    ],
    "NP1": [
        "NN1",
        "NN1 who was ADJ",
        "NN1 who was not ADJ",
        "NN1 who was not ever ADJ",
        "NN1 who VB-intrans",
        "NN1 who DT NN1 VB-trans",
        "NN1 who DT NN1 VB-trans",
        "NN1 who no NN1 VB-trans",
        "NN1 who no NN1 ever VB-trans",
    ],
    "NP-bad": [
        "NP1 who ever VB-intrans",
        "NP1 who was ever ADJ",
        "NN1 who DT NP-bad VB-trans",
    ],
    "NP-neg": ["NP1", "NN1 who ever VB-intrans", "NN1 who was ever ADJ"],
    "NN1": ["NN"],  # , 'NN prep', 'NN not prep'],
    # lexical items borrowed from Allyson Ettinger's paper
    # https://github.com/aetting/compeval-generation-system/blob/master/lexical/vocabulary.json
    "NN": [
        "professor",
        "student",
        "man",
        "woman",
        "president",
        "child",
        "girl",
        "boy",
        "judge",
        "senator",
        "secretary",
        "doctor",
        "lawyer",
        "scientist",
        "banker",
        "assistant",
        "officer",
    ],
    "NN-plural": [
        "professors",
        "students",
        "men",
        "women",
        "presidents",
        "children",
        "girls",
        "boys",
        "judges",
        "senators",
        "secretaries",
        "doctors",
        "lawyers",
        "scientists",
        "bankers",
        "assistants",
        "officers"
    ],
    "prep": [
        "in the room",
        "at home",
        "on a run",
        "under the tree",
        "in the car",
        "on the bridge",
        "at work",
        "at the park",
        "with the group",
    ],
    "VB-trans": [
        "thanked",
        "pushed",
        "tricked",
        "hugged",
        "recommended",
        "called",
        "followed",
        "helped",
        "supported",
        "watched",
        "contacted",
        "hit",
        "met",
        "hated",
        "liked",
        "believed",
        "loved",
        "observed",
        "avoided",
        "advised",
    ],
    "VB-trans-present": [
        "thanks",
        "pushes",
        "tricks",
        "hugs",
        "recommends",
        "calls",
        "follows",
        "helps",
        "supports",
        "watches",
        "contacts",
        "hits",
        "meets",
        "hates",
        "likes",
        "believes",
        "loves",
        "observes",
        "avoids",
        "advises",
    ],
    "VB-intrans": [
        "succeeded",
        "failed",
        "traveled",
        "smiled",
        "slept",
        "danced",
        "ran",
        "shouted",
        "resigned",
    ],
    "VB-intrans-present": [
        "succeeds",
        "fails",
        "travels",
        "smiles",
        "sleeps",
        "dances",
        "runs",
        "shouts",
        "resigns",
    ],
    "ADJ": ["smart", "funny", "happy", "sad", "right", "wrong"],
    "DT": ["the", "some"], #["a", "the", "some"],
}

def generate_wrapper(config):
    '''Expects a dictionary with the following keys:
       - section (str)
       - licensed (0/1)
       - negation (0/1)
       - long (0/1)
       - present_tense (0/1)
       - singular (0/1)
       nan indicates that there's no preference.'''
    if config["licensed"] == 0:
        result = generate("S-bad", config)
    else:
        result = generate("S-good", config)

    if "ever" not in result:
        return generate_wrapper(config)

    if config["negation"] == 0:
        if "no" in result or "not" in result:
            return generate_wrapper(config)
    else:
        if not("no" in result or "not" in result):
            return generate_wrapper(config)

    # if None, don't do anything
    if config["long"] == 0:
        if len(result.split()) > 15:
            return generate_wrapper(config)
    elif config["long"] == 1:
        if len(result.split()) <= 15:
            return generate_wrapper(config)

    return result
    

def generate(tpl, config):
    '''Expects a dictionary with the following keys:
       - section (str)
       - licensed (0/1/nan)
       - negation (0/1/nan)
       - long (0/1/nan)
       - present_tense (0/1/nan)
       - singular (0/1/nan)
       nan indicates that there's have no preference.'''

    if config["present_tense"] == 1:
        tpl = tpl.replace("VB-trans", "VB-trans-present")
        tpl = tpl.replace("VB-intrans", "VB-intrans-present")
        tpl = tpl.replace("was", "is")
    
    if config["singular"] == 0:
        # NOTE: we need the spaces to make sure we don't replace the NN in NN1 for example
        tpl = tpl.replace("NN ", "NN-plural ")
        tpl = tpl.replace("was", "were")

    toks = []
    for t in tpl.split():
        # NOTE: the present tense verbs are all singular, so the templates won't produce good
        # sentences in present tense and singular
        if t in grammar:
            toks.append(random.choice(grammar[t]))
        else:
            toks.append(t)
    new = " ".join(toks)
    if not new == tpl:
        # print(new)
        return generate(new, config)
    return new + " ."


def jsonify(sent, label, co_occurs, section):
    return {
        "sentence": sent,
        "label": label,
        "co-occurs": co_occurs,
        "section": section,
    }


def make_dataset(
    both_json_copy,
    neither_json_copy,
    weak_only_json_copy,
    both_count,
    neither_count,
    weak_only_count,
    flip_weak_only=False,
):
    both_els = both_json_copy[:both_count]
    del both_json_copy[:both_count]

    neither_els = neither_json_copy[:neither_count]
    del neither_json_copy[:neither_count]

    weak_only_els = weak_only_json_copy[:weak_only_count]
    if flip_weak_only:
        for ex in weak_only_els:
            ex["label"] = 1

    del weak_only_json_copy[:weak_only_count]

    return both_els + neither_els + weak_only_els


def make_tsv_line(el):
    return "{}\t{}\t{}\t{}\n".format(
        el["sentence"], el["section"], el["co-occurs"], el["label"]
    )

@plac.opt(
    "weak", "weak feature to use", choices=["tense", "lexical", "length", "plural"]
)
def main(weak="lexical"):
    random.seed(42)

    config_path = os.path.join("data/npi", f"{weak}.csv")
    section_to_configs = properties.get_config(config_path)
    section_to_examples = {"both": [], "neither": [], "weak": []}

    for section in section_to_examples:
        # NOTE: there's only one config per section, so we'll just take that one
        config = section_to_configs[section][0]

        for _ in range(5_000):
            sentence = generate_wrapper(config)
            section_to_examples[section].append(sentence)

    both_json = [jsonify(sent, 1, True, "both") for sent in section_to_examples["both"]]
    neither_json = [jsonify(sent, 0, True, "neither") for sent in section_to_examples["neither"]]
    weak_only_json = [jsonify(sent, 0, False, "weak") for sent in section_to_examples["weak"]]

    if not os.path.exists("./properties"):
        os.mkdir("./properties")
    if not os.path.exists(f"./properties/npi_{weak}/"):
        os.mkdir(f"./properties/npi_{weak}/")

    # Use shared API to generate datasets as a function of the rate.
    base_df = pd.concat(
        [pd.DataFrame(both_json), pd.DataFrame(neither_json)]
    ).drop_duplicates()
    base_df["prop"] = "npi"
    train_base, test_base = train_test_split(base_df, test_size=0.5)
    counterexample_df = pd.DataFrame(weak_only_json).drop_duplicates()
    counterexample_df["prop"] = "npi"
    train_counterexample, test_counterexample = train_test_split(
        counterexample_df, test_size=0.5
    )
    rates = [0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
    properties.generate_property_data(
        "npi_{}".format(weak),
        "weak",
        train_base,
        test_base,
        train_counterexample,
        test_counterexample,
        1000,
        rates,
    )


if __name__ == "__main__":
    plac.call(main)
