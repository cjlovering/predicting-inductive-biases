import argparse
import math
import random
import os
import numpy as np
import pandas as pd
import torch
import math
import os
import random
import numpy as np
from sklearn.model_selection import train_test_split


import properties


def get_parser():
    parser = argparse.ArgumentParser(
        description="Runs a small model for experimenting with rule acquisition"
    )

    # General parameters
    parser.add_argument(
        "--data",
        type=str,
        default="./properties/",
        help="directory to store data files and models",
    )
    parser.add_argument(
        "--device",
        help="pass --device cuda to run on gpu. if you select cuda when no cuda is availabe, it will break.",
        default="cuda",
    )
    parser.add_argument(
        "--num_loops",
        type=int,
        default=1,
        help="number of times to run the whole training loop to convergence",
    )

    # Parameters for data generation
    parser.add_argument(
        "--num_counter_examples",
        type=int,
        default=50,
        help="number of examples for which the disctractor property will lead the model astray",
    )
    parser.add_argument(
        "--label_split",
        type=float,
        default=0.5,
        help="proportion of examples to have the label 0 (the label for which the true property does not hold)",
    )
    parser.add_argument("--vocab_size", type=int, default=50_000)
    parser.add_argument("--train_size", type=int, default=100_000)
    parser.add_argument("--seq_length", type=int, default=10)
    parser.add_argument("--initial_true_only_examples", type=int, default=0)
    parser.add_argument(
        "--true_property",
        type=int,
        help="which true property to use out of {1,2,3,4,5}",
    )
    parser.add_argument("--hold_out", action="store_true")
    parser.add_argument(
        "--num_distractors", type=int, default=1, help="number of distractor properties"
    )
    parser.add_argument(
        "--num_unremovable_distractors",
        type=int,
        default=0,
        help="number of distractor properties for which we cannnot generate case #4 counter-examples",
    )
    parser.add_argument("--experiment_id", type=str, default=None)
    parser.add_argument("--rand_seed", type=int, default=42)
    parser.add_argument(
        "--sample_zipfian",
        action="store_true",
        help="If true, the symbols will follow a zipfian distribution",
    )
    return parser


"""
Deals with building the data file (and returning corpora) and contains a couple small utilities.
A lot of detail in the comment for the make_data function.
"""


class DataHandler:
    def __init__(
        self,
        data_path,
        label_split,
        rate,
        vocab_size,
        seq_length,
        true_property,
        hold_out,
        experiment_id,
        num_distractors,
        num_unremovable_distractors,
        initial_true_only_examples,
        sample_zipfian: bool,
    ):
        self.data_path = data_path
        self.label_split = label_split
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.true_property = true_property
        self.hold_out = hold_out
        self.num_distractors = num_distractors
        self.num_unremovable_distractors = num_unremovable_distractors
        self.initial_true_only_examples = initial_true_only_examples
        self.sample_zipfian = sample_zipfian  # bool flag

        # Makes the data directory

        # true property will be a list of 1.
        self.data_dir = f"./properties/toy_{args.true_property}"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def has_adjacent_duplicate(self, sent):
        for i in range(len(sent) - 1):
            if sent[i + 1] == sent[i]:
                return True
        return False

    def has_first_and_last_duplicate(self, sent):
        return sent[0] == sent[len(sent) - 1]

    def get_random_sent(self, white_list, black_list, shorten_sent: int, is_test: bool):
        """Returns a sentence of length self.seq_length - shorten_sent with tokens from [0, vocab_size - 1].
        Guaranteed not to include anything from black_list and to include anything from white_list
        exactly once."""
        white_set = set(white_list)
        black_set = set(black_list).union(white_list)

        sent_len = self.seq_length - shorten_sent - len(white_list)
        sent_clean = False
        while not sent_clean:
            sent = []
            for _ in range(sent_len):
                # add new token -- this will add heldout tokens at test time.
                sent.append(str(self.get_new_token(white_list, black_list, is_test)))

            sent_clean = True
            for black_listed_number in black_set:
                if str(black_listed_number) in sent:
                    sent_clean = False
                    continue

            for white_listed_number in white_set:
                white_listed_number_index = random.randint(0, len(sent))
                sent.insert(white_listed_number_index, str(white_listed_number))

        return sent

    def get_white_list(self, distractor_prop, case, test):
        # NOTE: this isn't very clear, but this will happen if we're building a classification dataset
        # TODO: make this cleaner
        if distractor_prop is None:
            return []

        if not distractor_prop and not (
            case == 4 and self.num_unremovable_distractors > 0 and not test
        ):
            return []

        # if num_distractors is two, this will give equal probability to [2], [3], and [2, 3]
        # if num_distractors is three, this will give equal probability to [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]
        white_list = []
        while len(white_list) == 0:
            for i in range(2, 2 + self.num_distractors):
                if random.randint(0, 1) == 1:
                    white_list.append(i)

        if case == 4 and self.num_unremovable_distractors > 0:
            remove_list = list(range(2, 2 + self.num_distractors))
            for i in range(self.num_unremovable_distractors):
                remove_index = random.randint(0, len(remove_list) - 1)
                remove_list.pop(remove_index)

            for i in remove_list:
                if i in white_list:
                    white_list.remove(i)

        return white_list

    def get_black_list(self, distractor_prop, case, test):
        # NOTE: this isn't very clear, but this will happen if we're building a classification dataset
        # TODO: make this cleaner
        if distractor_prop is None:
            return []

        if not distractor_prop:
            if not (case == 4 and self.num_unremovable_distractors > 0 and not test):
                return list(range(2, 2 + self.num_distractors))
            else:
                return list(
                    range(
                        2 + self.num_unremovable_distractors, 2 + self.num_distractors
                    )
                )

        return []

    def get_new_token(self, white_list, black_list, test):
        """Returns a random new token that's in neither white_list nor black_list. The token
        will be sampled from disjoint sets for test=True/False if self.hold_out."""
        if self.hold_out:
            if not test:
                low = 0
                high = math.floor(0.75 * self.vocab_size)
            else:
                low = math.floor(0.75 * self.vocab_size) + 1
                high = self.vocab_size - 1
        else:
            low = 0
            high = self.vocab_size - 1

        new_token = self.sample_token(low, high)
        while (
            new_token in black_list or new_token in white_list
        ):  # tokens in the white-list should only be included once
            new_token = self.sample_token(low, high)
        return new_token

    def sample_token(self, low, high):
        """Samples a token between the low & high symbol values.

        We implicitly assume that the symbols index into an embedding
        and are not used for math or anything where the value itself
        has meaning.
        """
        if self.sample_zipfian:
            return self.sample_token_zipfian(low, high)
        else:
            return self.sample_token_uniform(low, high)

    def sample_token_uniform(self, low, high):
        new_token = random.randint(low, high)
        return new_token

    def sample_token_zipfian(self, low, high, a=1.5):
        """Samples from low to high in a zipfian

        # `a, a > 1` controls the flatness of the distribution. The lower
        # `a`, the flatter the distribution.
        """
        new_token = np.random.zipf(a=a) - 1
        while new_token < high - low:
            new_token = np.random.zipf(a=a) - 1
        new_token = high - new_token
        assert low <= new_token
        assert new_token < high
        return new_token

    # 1: true
    def get_one(self, distractor_prop, test, case):
        # NOTE: we need the extra parameters for this function to be used like any other
        """
        Positive Examples
        -----------------
        10 19 1 14 10
        1 22 11 11 97

        Negative Examples
        -----------------
        11 11 13 14 15
        11 12 11 14 15
        """
        sent = self.get_random_sent(
            self.get_white_list(distractor_prop, case, test) + [1],
            self.get_black_list(distractor_prop, case, test),
            0,
            test,
        )
        return sent

    # 2: true
    def get_first_and_last_duplicate(self, distractor_prop, test, case):
        """
        Positive Examples
        -----------------
        10 19 12 14 10
        97 22 11 11 97

        Negative Examples
        -----------------
        11 11 13 14 15
        11 12 11 14 15
        """
        black_list = self.get_black_list(distractor_prop, case, test)
        white_list = self.get_white_list(distractor_prop, case, test)
        sent = self.get_random_sent(white_list, black_list, 2, test)
        new_token = self.get_new_token(white_list, black_list, test)

        sent = [str(new_token)] + sent + [str(new_token)]
        return sent

    # 3: true
    def get_prefix_duplicate(self, distractor_prop, test, case):
        """This is a function checks if the first two items in the list are duplicates.

        Positive Examples
        -----------------
        2 2 1 2 0
        1 1 3 1 1
        0 0 0 0 0

        Negative Examples
        -----------------
        1 0 0 0 0
        0 1 1 1 1
        0 1 1 2 2
        """
        black_list = self.get_black_list(distractor_prop, case, test)
        white_list = self.get_white_list(distractor_prop, case, test)
        sent = self.get_random_sent(white_list, black_list, 2, test)
        new_token = self.get_new_token(white_list, black_list, test)

        sent = [str(new_token), str(new_token)] + sent
        return sent

    # 4: true
    def get_contains_first(self, distractor_prop, test, case):
        """This is a function of the first item in the list:

        Positive Examples
        -----------------
        0 1 1 2 0
        0 0 1 1 0
        0 0 0 0 0

        Negative Examples
        -----------------
        1 0 0 0 0
        0 1 1 1 1
        0 1 1 2 2
        """
        black_list = self.get_black_list(distractor_prop, case, test)
        white_list = self.get_white_list(distractor_prop, case, test)
        sent = self.get_random_sent(white_list, black_list, 2, test)
        new_token = self.get_new_token(white_list, black_list, test)

        sent.insert(0, str(new_token))
        sent.insert(random.randint(0, len(sent)), str(new_token))
        return sent

    # 5: true
    def get_duplicate(self, distractor_prop, test, case):
        """
        Positive Examples
        -----------------
        10 10 12 14 19
        15 22 11 11 97
        90 12 12 15 20

        Negative Examples
        -----------------
        11 12 13 14 15
        11 12 11 14 15
        """

        black_list = self.get_black_list(distractor_prop, case, test)
        white_list = self.get_white_list(distractor_prop, case, test)
        sent = self.get_random_sent(white_list, black_list, 2, test)
        new_token = self.get_new_token(white_list, black_list, test)

        new_token_index = random.randint(0, len(sent))
        sent.insert(new_token_index, str(new_token))
        sent.insert(new_token_index, str(new_token))
        return sent

    # true
    def get_with_props(self, distractor_prop, test, get_props, case):
        get_prop = get_props[random.randint(0, len(get_props) - 1)]
        return get_prop(distractor_prop, test, case)

    # not true
    def get_without_props(self, distractor_prop, test, has_prop_checkers, case):
        sent = self.get_random_sent(
            self.get_white_list(distractor_prop, case, test),
            self.get_black_list(distractor_prop, case, test),
            0,
            test,
        )
        while any(
            [has_prop_checker(sent) for has_prop_checker in has_prop_checkers]
        ):  # if one of these is True we want to sample again
            sent = self.get_random_sent(
                self.get_white_list(distractor_prop, case, test),
                self.get_black_list(distractor_prop, case, test),
                0,
                test,
            )

        return sent

    def get_get_props(self):
        get_props = []
        has_prop_checkers = []

        # Returns functions to generate examples based on self.true_property
        true_property = self.true_property
        if true_property == 1:
            get_prop = self.get_one
            has_prop_checker = lambda sent: "1" in sent
        elif true_property == 2:
            get_prop = self.get_first_and_last_duplicate
            has_prop_checker = self.has_first_and_last_duplicate
        elif true_property == 3:
            get_prop = self.get_prefix_duplicate
            has_prop_checker = lambda sent: sent[0] == sent[1]
        elif true_property == 4:
            get_prop = self.get_contains_first
            has_prop_checker = lambda sent: any(sent[0] == w for w in sent[1:])
        elif true_property == 5:
            get_prop = self.get_duplicate
            has_prop_checker = self.has_adjacent_duplicate
        else:
            raise NotImplementedError("True property hasn't been implemented yet.")

        get_props.append(get_prop)
        has_prop_checkers.append(has_prop_checker)

        return (get_props, has_prop_checkers)

    def make_data(
        self, corpus_path, weak_size, both_size, neither_size, strong_size, test
    ):
        """Returns a Corpus with corpus_size examples.

        All sentence lengths will be self.seq_length. The data files will be placed in a randomly named
        directory, with each example on a line as follows: "{example}|{label}|{case}"

        Case I: property holds
        Case II: property doesn't hold

        The distractor property is the presence of a two. The true properties (and the specifications for the absence) are
        above. The 'property' will be the distractor property if self.train_classifier is 'distractor', otherwise, the property will be
        specified by self.true_property.
        """
        get_trues, true_checkers = self.get_get_props()
        out = []
        with open(corpus_path, "w") as f:
            f.write("sentence\tlabel\tsection\n")

            # Case I
            for _ in range(weak_size):
                # Distractor but not true
                sent = self.get_without_props(True, test, true_checkers, 1)
                out.append({"sentence": " ".join(sent), "label": 0, "section": "weak"})

            # Case II
            for _ in range(both_size):
                # Distractor and true
                sent = self.get_with_props(True, test, get_trues, 2)
                out.append({"sentence": " ".join(sent), "label": 1, "section": "both"})

            # Case III
            for _ in range(neither_size):
                # Neither distractor, nor true
                sent = self.get_without_props(False, test, true_checkers, 3)
                out.append(
                    {"sentence": " ".join(sent), "label": 0, "section": "neither"}
                )

            # Case IV
            for _ in range(strong_size):
                # True but not distractor
                sent = self.get_with_props(False, test, get_trues, 4)
                out.append(
                    {"sentence": " ".join(sent), "label": 1, "section": "strong"}
                )

        data = pd.DataFrame(out)
        return data


def main(args):
    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    data_handler = DataHandler(
        args.data,
        args.label_split,
        args.num_counter_examples,
        args.vocab_size,
        args.seq_length,
        args.true_property,
        args.hold_out,
        args.experiment_id,
        args.num_distractors,
        args.num_unremovable_distractors,
        args.initial_true_only_examples,
        args.sample_zipfian,
    )
    data = data_handler.make_data(
        f"{data_handler.data_path}/all.tsv",
        weak_size=105_000,
        both_size=105_000,
        neither_size=105_000,
        strong_size=105_000,
        test=False,
    )
    rates = [0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
    train_base, test_base = train_test_split(
        data[(data.section != "weak") & (data.section != "strong")], test_size=5000
    )
    train_counterexample, test_counterexample = train_test_split(
        data[data.section == "weak"], test_size=1000
    )
    train_counterexample_strong, test_counterexample_strong = train_test_split(
        data[data.section == "strong"], test_size=1000
    )
    test_counterexample = pd.concat([test_counterexample, test_counterexample_strong])
    properties.generate_property_data(
        "toy_{}".format(args.true_property),
        "weak",
        train_base,
        test_base,
        train_counterexample,
        test_counterexample,
        100_000,
        rates,
        test_section_size=1000,
    )
    properties.generate_property_data_strong_direct(
        "toy_{}".format(args.true_property),
        "weak",
        train_base,
        test_base,
        train_counterexample_strong,
        test_counterexample_strong,
        100_000,
        rates,
        test_section_size=1000,
    )


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
