import os
import random

import nltk
import pandas as pd
import plac
from nltk.corpus import verbnet as vn
from sklearn.model_selection import train_test_split
import pdb

import properties
import numpy as np

nltk.download("verbnet")
relations = [
    "sister",
    "brother",
    "daughter",
    "son",
    "mother",
    "father",
    "cousin",
    "niece",
    "nephew",
    "grandmother",
    "grandfather",
    "grandson",
    "friend",
    "granddaughter",
    "boss",
    "employee",
    "supervisor",
    "mentor",
    "mentee",
    "teacher",
    "student",
    "French teacher",
    "piano teacher",
    "tutor",
    "plumber",
    "electrician",
    "handyman",
    "contractor",
    "hairdresser",
    "senator",
    "lawyer",
    "partner",
    "associate",
    "doctor",
    "dermatologist",
    "dentist",
    "oncologist",
    "podiatrist",
    "guest",
    "spouse",
    "wife",
    "husband",
    "boyfriend",
    "girlfriend",
    "ex-girlfriend",
    "ex-boyfriend",
    "ex-wife",
    "ex-husband",
    "best friend",
    "classmate",
    "colleague",
]
time = ["often", "sometimes", "rarely", "occasionally"]


def pluralize(word):
    if word[-1] == "y" and word[-2] != "o" and word[-2] != "a":
        return word[0:-1] + "ies"
    elif word[-1] == "x" or word[-1] == "s" or word[-1] == "h":
        return word + "es"
    elif word.endswith("man"):
        return word[0:-2] + "en"
    elif word.endswith("fe"):
        return word[0:-2] + "ves"
    else:
        return word + "s"


def get_template(config):
    """Expects a dictionary with the following keys:
       - section (str)
       - subject_singular (0/1/nan)
       - closest_noun_singular (0/1/nan)
       - verb_singular (0/1/nan)
       - time_word (0/1/nan)
       - loops (0/1/nan): 0 means no loops, 1 means at least one, and nan means any from 0-infinity 
       nan indicates that there's have no preference."""
    sent = "beginning subject of the loops closest-noun verb the object"

    time_word = config["time_word"]
    if time_word == 0:
        sent = sent.replace("beginning", "the")
    elif time_word == 1:
        sent = sent.replace("beginning", "time the")

    subject_singular = config["subject_singular"]
    if subject_singular == 0:
        sent = sent.replace("subject", "relation-plural")
    elif subject_singular == 1:
        sent = sent.replace("subject", "relation-singular")
    else:
        sent = sent.replace("subject", "relation")

    closest_noun_singular = config["closest_noun_singular"]
    if closest_noun_singular == 0:
        sent = sent.replace("closest-noun", "relation-plural")
    elif closest_noun_singular == 1:
        sent = sent.replace("closest-noun", "relation-singular")
    else:
        sent = sent.replace("closest-noun", "relation")

    verb_singular = config["verb_singular"]
    if verb_singular == 0:
        sent = sent.replace("verb", "verb-plural")
    elif verb_singular == 1:
        sent = sent.replace("verb", "verb-singular")

    loops = config["loops"]
    if loops == 0:
        sent = sent.replace("loops", "")
    elif loops == 1:
        sent = sent.replace("loops", "loops-1")

    sent = sent.replace("object", "relation")
    return " ".join(sent.split())


grammar = {
    # should be pluralizable
    # should be able to say "<relation-singular> of the guy"
    "relation-singular": relations,
    "relation": ["relation-singular", "relation-plural"],
    # should be able to say "they <verb-plural> me"
    "verb-plural": vn.lemmas("admire-31.2") + vn.lemmas("amuse-31.1"),
    "verb": ["verb-singular", "verb-plural"],
    "time": time,
    "beginning": ["time the", "the"],
    "loops": ["", "relation of the loops"],
    "loops-1": ["relation of the loops"],
}

grammar["relation-plural"] = [
    pluralize(relation) for relation in grammar["relation-singular"]
]
grammar["verb-singular"] = [pluralize(verb) for verb in grammar["verb-plural"]]


def generate(tpl):
    toks = []
    for t in tpl.split():
        if t in grammar:
            toks.append(random.choice(grammar[t]))
        else:
            toks.append(t)
    new = " ".join(toks)
    if not new == tpl:
        # print(new)
        return generate(new)
    return new + " ."


def make_dataset(section_to_count, template, easy_feature):
    dataset = []

    config_path = os.path.join("data/sva", f"{template}_{easy_feature}.csv")
    section_to_configs = properties.get_config(config_path)

    for section in section_to_count:
        templates = []
        for config in section_to_configs[section]:
            templates.append(get_template(config))

        for _ in range(section_to_count[section]):
            sentence = generate(random.choice(templates))
            label = 1 if section == "both" or section == "strong" else 0
            dataset.append({"sentence": sentence, "label": label, "section": section})
    return dataset


def make_tsv_line(el):
    return "{}\t{}\t{}\n".format(el["sentence"], el["section"], el["label"])


@plac.opt("template", "template to use", choices=["base", "hard"])
@plac.opt(
    "weak", "weak feature to use", choices=["agreement", "lexical", "length", "plural"]
)
def main(template="base", weak="lexical"):
    random.seed(42)
    section_size = 1000
    if not os.path.exists("./properties"):
        os.mkdir("./properties")
    if not os.path.exists(f"./properties/sva_{template}_{weak}/"):
        os.mkdir(f"./properties/sva_{template}_{weak}/")

    dataset = make_dataset(
        # 1250 to handle duplicates.
        {
            "both": section_size + 1250,
            "neither": section_size + 1250,
            "weak": section_size + 1250,
            "strong": 0 * (section_size + 250),
        },
        template,
        weak,
    )
    all_df = pd.DataFrame(dataset).drop_duplicates()

    base_df = all_df[all_df.section.isin({"both", "neither"})]
    train_base, test_base = train_test_split(base_df, test_size=0.5)

    counterexample_df = all_df[all_df.section.isin({"weak"})]
    train_counterexample, test_counterexample = train_test_split(
        counterexample_df, test_size=0.5
    )
    rates = [0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
    properties.generate_property_data(
        "sva_{}_{}".format(template, weak),
        "weak",
        train_base,
        test_base,
        train_counterexample,
        test_counterexample,
        section_size,
        rates,
    )


if __name__ == "__main__":
    plac.call(main)
