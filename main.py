import itertools
import random
import os
import numpy as np
import pandas as pd
import plac
import pytorch_lightning as pl
import sklearn.metrics as metrics
import torch
import tqdm

from pytorch_lightning import Trainer

from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.callback import Callback

from models import bert, lstm_glove, lstm_toy, roberta, t5, gpt2


@plac.opt(
    "prop",
    "property name",
    choices=[
        "gap-base-length",
        "gap-base-plural",
        "gap-hard-length",
        "gap-hard-none",
        "gap-hard-tense",
        "gap-base-lexical",
        "gap-base-tense",
        "gap-hard-lexical",
        "gap-hard-plural",
        "npi_lexical",
        "npi_plural",
        "npi_tense",
        "npi_length",
        "sva_base_agreement",
        "sva_base_lexical",
        "sva_base_plural",
        "sva_hard_agreement",
        "sva_hard_lexical",
        "sva_hard_length",
        "sva_hard_plural",
        "toy_1",
        "toy_2",
        "toy_3",
        "toy_4",
        "toy_5",
    ],
)
@plac.opt(
    "rate",
    type=float,
    help=(
        "This is the co-occurence rate between the counter examples and the labels"
        "We generate data for rates {0., 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0}."
        "We use a rate=-1. when the task is `probing` as a filler value"
        "but its not used or checked, so anything is fine."
    ),
)
@plac.opt(
    "probe",
    "probing feature",
    choices=["strong", "weak", "n/a", "strong_direct", "msgs"],
    abbrev="prb",
)
@plac.opt("task", "which mode/task we're doing", choices=["probing", "finetune"])
@plac.opt(
    "model",
    "which model to use; use a hugging face model.",
)
@plac.opt(
    "seed",
    "which rand seed to use",
    type=int,
)
@plac.opt(
    "wandb_entity", "wandb entity. set WANDB_API_KEY (in script or bashrc) to use."
)
def main(
    prop="sva",
    rate=0,
    probe="strong",
    task="finetune",
    model="bert-base-uncased",
    seed=1,
    wandb_entity="bert-syntax",
):
    """Trains and evaluates model.

    NOTE:
    * If `task` = finetune, then `probe` is ignored.
    * If `task` = probe, then `rate` is ignored.

    NOTE: Use the `properties.py` file to generate your data.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using: {accelerator}.")
    batch_size = 128
    accumulate_grad_batches = 1

    if "t5" in model:
        # t5 uses more memory. TODO: Aggregate gradients.
        batch_size = 64
        accumulate_grad_batches = 2

    if "gpt" in model:
        batch_size = 64

    # Lower the following to (1, 0.1, 0.1) to speed up debugging.
    if "toy" in prop:
        # toy props has more data - less epochs needed.
        num_epochs = 10
    else:
        # NOTE(Fall 2022): Originally did 50 epochs;
        # This could probably be reduced and/or early stopping added.
        # There is some issue with adding early stopping if you're interested
        # in the LossAuc.
        num_epochs = 50

    limit_train_batches = 1.0
    limit_test_batches = 1.0

    ## constants
    if task == "finetune":
        # TODO: Fix elsewhere.
        if rate == 0:
            rate = int(0)
        title = f"{prop}_{task}_{rate}_{model}_{seed}"
        path = f"{task}_{rate}"
    else:
        title = f"{prop}_{task}_{probe}_{model}_{seed}"
        path = f"{task}_{probe}"
    title = title.replace("/", "_")
    if os.path.exists(f"results/stats/{title}.tsv"):
        exit(f"Ending job: result exists already: {title}")

    # We use huggingface for transformer-based models and spacy for baseline models.
    # The models/pipelines use slightly different APIs.
    negative_label = 0
    positive_label = 1

    if "t5" in model:
        # use "True" / "False"
        label_col = "label"
    else:
        # use 0, 1
        label_col = "label"

    # NOTE: Set `entity` to your wandb username, and add a line
    # to your `.bashrc` (or whatever) exporting your wandb key.
    # `export WANDB_API_KEY=62831853071795864769252867665590057683943`.
    config = dict(prop=prop, rate=rate, probe=probe, task=task, model=model, seed=seed)

    wandb_logger = WandbLogger(entity=wandb_entity, project="features")
    wandb_logger.log_hyperparams(config)
    train_data, eval_data, test_data = load_data(
        prop, path, label_col, [positive_label, negative_label]
    )
    num_steps = (len(train_data) // batch_size) * num_epochs
    datamodule = DataModule(batch_size, train_data, eval_data, test_data)

    # Check ~10% of the validation data every 1/10 epoch.
    # We shuffle the validation data so we get new examples.
    limit_val_batches = max(0.1, 1 / len(datamodule.val_dataloader()))
    val_check_interval = max(0.1, 1 / len(datamodule.val_dataloader()))

    classifier = load_model(model, num_steps)
    lossauc = LossAuc()
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        logger=wandb_logger,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        val_check_interval=val_check_interval,
        # early_stop_callback=False,
        min_epochs=num_epochs,
        max_epochs=num_epochs,
        callbacks=[lossauc],
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer.fit(classifier, datamodule)

    # Test
    test_result = trainer.test(datamodule=datamodule)[0]
    classifier.freeze()
    classifier.eval()
    with torch.no_grad():
        test_pred = []
        for batch in datamodule.test_dataloader():
            logits = classifier(batch)
            test_pred.extend(logits.argmax(1).cpu().numpy())

    test_df = pd.read_table(f"./properties/{prop}/test.tsv")
    test_df["pred"] = test_pred
    test_df.to_csv(
        f"results/raw/{title}.tsv",
        sep="\t",
        index=False,
    )

    # Additional evaluation.
    if task == "finetune":
        additional_results = finetune_evaluation(test_df, label_col)
    elif task == "probing":
        additional_results, block_logs = compute_mdl(
            train_data, model, batch_size, num_epochs, accumulate_grad_batches
        )
        block_logs_df = pd.DataFrame(block_logs)
        # block_logs_df["section"] = (est_df.section.iloc[0],)
        for k, v in config.items():
            block_logs_df[k] = v
        block_logs_df.to_csv(
            f"./results/raw/block-{title}.tsv",
            sep="\t",
            index=False,
        )
    else:
        # For the toy data, this takes SO long. I have to look into it.
        # This seems to be a bigger problem with the lstms...
        additional_results = {}

    pd.DataFrame(
        [
            {
                # NOTE: `loss_auc` is not tracked when finetuning.
                "val_loss_auc": lossauc.get(),
                **test_result,
                **additional_results,
                **config,  # log results for easy post processing in pandas, etc.
                "section": test_df.section.iloc[0],
            }
        ]
    ).to_csv(
        f"./results/stats/{title}.tsv",
        sep="\t",
        index=False,
    )


def load_data(prop, path, label_col, categories):
    """Load data from the IMDB dataset, splitting off a held-out set."""
    # SHUFFLE
    trn = (
        pd.read_table(f"./properties/{prop}/{path}_train.tsv")
        .sample(frac=1)
        .reset_index(drop=True)
    )
    val = (
        pd.read_table(f"./properties/{prop}/{path}_val.tsv")
        .sample(frac=1)
        .reset_index(drop=True)
    )
    # NO SHUFFLE (so we can re-align the results with the input data.)
    tst = pd.read_table(f"./properties/{prop}/test.tsv")

    # SPLIT & PREPARE
    trn_txt, trn_lbl = (trn.sentence.tolist(), trn[label_col].tolist())
    val_txt, val_lbl = (val.sentence.tolist(), val[label_col].tolist())
    tst_txt, tst_lbl = (tst.sentence.tolist(), tst[label_col].tolist())

    train_data = list(zip(trn_txt, trn_lbl))
    eval_data = list(zip(val_txt, val_lbl))
    test_data = list(zip(tst_txt, tst_lbl))
    print("train", len(train_data))
    print("val", len(eval_data))
    print("test", len(test_data))
    return train_data, eval_data, test_data


def load_model(model, num_steps):
    """Loads appropriate model & optimizer (& optionally lr scheduler.)

    Parameters
    ----------
    model : ``str``
        model string. in most cases, a hugging face model code.
    num_steps : ``int``
        number of update steps. optionally used for lr schedules.
    """
    if "gpt2" in model:
        return gpt2.GPT2Classifier(model, num_steps)
    if "roberta" in model:
        return roberta.RobertaClassifier(model, num_steps)
    if "bert" in model:
        return bert.BertClassifier(model, num_steps)
    if "t5" in model:
        return t5.T5Classifier(model, num_steps)
    if "lstm-glove" in model:
        return lstm_glove.LstmGloveClassifier(model)
    if "lstm-toy" in model:
        return lstm_toy.LstmToyClassifier(model)

    assert f"model `{model}` not found."


def finetune_evaluation(df, label_col):
    """Compute additional evaluation.

    1. Use `label` for the label.
    2. Use `section` and denote which of `{weak, strong, both, neither} hold.
    """
    df["error"] = df["pred"] != df[label_col]
    # For "weak_feature", we mean the `weak_feature` is present in the example.
    df["weak_feature"] = ((df.section == "both") | (df.section == "weak")).astype(int)
    both = df[df.section == "both"]
    neither = df[df.section == "neither"]
    strong = df[df.section == "strong"]
    weak = df[df.section == "weak"]

    # Here we use `label` as 1:1 map for the strong feature. This might not hold up
    # if we move to using composite strong features.
    I_pred_true = metrics.mutual_info_score(df[label_col], df["pred"])
    I_pred_weak = metrics.mutual_info_score(df["weak_feature"], df["pred"])
    error = lambda x: x["error"].mean()
    score = lambda x: 1 - x["error"].mean()
    return {
        "test-error": error(df),
        "both-error": error(both),
        "neither-error": error(neither),
        "strong-error": error(strong),
        "weak-error": error(weak),
        "test-accuracy": score(df),
        "both-accuracy": score(both),
        "neither-accuracy": score(neither),
        "strong-accuracy": score(strong),
        "weak-accuracy": score(weak),
        "I-pred-true": I_pred_true,
        "I-pred-weak": I_pred_weak,
    }


def random_split_partition(zipped_list, sizes):
    # NOTE: I'm getting some strange issues where the 0.1% doesn't have
    # two labels, thus it gets some bad errors. 0.1% = 0.001, for 2000 * 0.001 = 2,
    # so fair enough.
    # SOLUTION: The training data is shuffled and contains equal counts (or close enough)
    # of labels.
    random.shuffle(zipped_list)
    pos = [z for z in zipped_list if z[1] in {1, "1", "yes"}]
    neg = [z for z in zipped_list if z[1] not in {1, "1", "yes"}]
    interleaved_list = list(itertools.chain(*zip(pos, neg)))
    return [
        interleaved_list[end - length : end]
        for end, length in zip(itertools.accumulate(sizes), sizes)
    ]


def compute_mdl(train_data, model, batch_size, num_epochs, accumulate_grad_batches):
    """Computes the Minimum Description Length (MDL) over the training data given the model.

    We use *prequential* MDL.

    Voita, Elena, and Ivan Titov. "Information-Theoretic Probing with Minimum Description Length."
    arXiv preprint arXiv:2003.12298 (2020). `https://arxiv.org/pdf/2003.12298`

    Parameters
    ----------
    ``train_data``: list of tuples of examples and labels.
    ``model``: A model string.
    """
    # NOTE: These aren't the split sizes, exactly; the first training size will be the first split size,
    # the second will be the concatenation of the first two, and so on. This is to take advantage
    # of the random_split function.
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    split_proportions = np.array(
        [0.1, 0.1, 0.2, 0.4, 0.8, 1.6, 3.05, 6.25, 12.5, 25, 50]
    )
    split_sizes = np.ceil(0.01 * len(train_data) * split_proportions)

    # How much did we overshoot by? We'll just take this from the longest split
    extra = np.sum(split_sizes) - len(train_data)
    split_sizes[len(split_proportions) - 1] -= extra

    splits = random_split_partition(train_data, split_sizes.astype(int).tolist())
    mdls = []
    block_logs = []

    # Cost to transmit the first via a uniform code
    mdls.append(split_sizes[0])

    for i in tqdm.trange(len(splits), desc="mdl"):
        # If training on the last block, we test on all the data.
        # Otherwise, we train on the next split.
        last_block = i == (len(splits) - 1)

        # setup the train and test split.
        train_split = list(itertools.chain.from_iterable(splits[0 : i + 1]))
        test_split = train_split if last_block else splits[i + 1]

        # re-fresh model.
        datamodule = DataModule(
            batch_size, train_split, test_split[:batch_size], test_split
        )
        num_steps = (len(train_split) // batch_size) * num_epochs
        classifier = load_model(model, num_steps)
        trainer = Trainer(
            accelerator=accelerator,
            devices=1,
            limit_train_batches=1.0,
            limit_val_batches=1.0,
            limit_test_batches=1.0,
            min_epochs=num_epochs,
            max_epochs=num_epochs,
            accumulate_grad_batches=accumulate_grad_batches,
        )
        trainer.fit(classifier, datamodule=datamodule)

        # Test
        test_result = trainer.test(datamodule=datamodule)
        test_loss = test_result[0]["test_loss"]
        block_logs.append(
            {
                "length": len(test_split),
                "loss": test_loss,
            }
        )

        if not last_block:
            mdls.append(test_loss)

    total_mdl = np.sum(np.asarray(mdls))
    # the last test_loss is of the model trained and evaluated on the whole training data,
    # which is interpreted as the data_cost
    data_cost = test_loss
    model_cost = total_mdl - data_cost
    return (
        {
            "total_mdl": total_mdl,
            "data_cost": data_cost,
            "model_cost": model_cost,
        },
        block_logs,
    )


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_data, eval_data, test_data):
        super().__init__()
        self.train_data = train_data
        self.eval_data = eval_data
        self.test_data = test_data
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.eval_data, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)


class LossAuc(Callback):
    def __init__(self, monitor="val_loss"):
        super().__init__()
        self.losses = []
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer, _):
        if trainer.sanity_checking:
            return
        logs = trainer.callback_metrics
        if self.monitor in logs:
            self.losses.append(logs[self.monitor])

    def get(self):
        if len(self.losses) == 0:
            return 0
        # We assume that the list contains pytorch tensor floats.
        return sum(self.losses).item()


if __name__ == "__main__":
    plac.call(main)
