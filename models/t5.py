import torch.nn as nn
import pytorch_lightning.metrics.functional as metrics
import torch
from transformers import (
    AdamW,
    T5Tokenizer,
    T5Model,
    get_linear_schedule_with_warmup,
)
import sklearn.metrics as sk_metrics
import pytorch_lightning as pl


class T5Classifier(pl.LightningModule):
    def __init__(self, model, num_steps, num_classes=2):
        super(T5Classifier, self).__init__()
        hidden_size = {"t5-small": 512, "t5-base": 768, "t5-large": 1024,}[model]
        self.model = T5Model.from_pretrained(model)
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        self.num_steps = num_steps
        self.classifier = nn.Linear(hidden_size, num_classes)

    def step(self, batch):
        """I was unable to get the model to *work* using the typical
        T5 text api. Here I try just getting the last hidden state
        and using a linear classifier on top of that.
        """
        texts, labels = batch
        texts = [format_input(t) for t in texts]
        input_ids = self.tokenizer.batch_encode_plus(
            texts, padding=True, return_tensors="pt", max_length=64
        )
        outputs = self.model(input_ids=input_ids["input_ids"])
        last_hidden_states = outputs[0][:, -1, :]
        logits = self.classifier(last_hidden_states)
        loss = nn.functional.cross_entropy(logits, labels)
        return logits, loss

    def forward(self, batch):
        """This is used for inference. """
        logits, _ = self.step(batch)
        return logits

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4,)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 0.1 * self.num_steps, self.num_steps
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # batch is tokenized.
        _, loss = self.step(batch)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        training_loss = sum([x["loss"] for x in outputs])
        return {"train_loss": training_loss, "log": {"train_loss": training_loss}}

    def validation_step(self, batch, batch_idx):
        # This is bad /:
        logits, loss = self.step(batch)
        _, labels = batch

        return {"val_loss": loss, "pred": logits.argmax(1), "true": labels}

    def validation_epoch_end(self, outputs):
        val_loss = sum([x["val_loss"] for x in outputs])
        pred = torch.cat([x["pred"] for x in outputs])
        true = torch.cat([x["true"] for x in outputs])
        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy(pred, true)
        # f_score = sk_metrics.f1_score(pred, true, average="macro")
        # accuracy = sk_metrics.accuracy_score(pred, true)
        out = {
            "val_loss": val_loss,
            "val_f_score": f_score,
            "val_accuracy": accuracy,
            "log": {
                "val_loss": val_loss,
                "val_f_score": f_score,
                "val_accuracy": accuracy,
            },
        }
        return out

    def test_step(self, batch, batch_idx):
        _, labels = batch
        logits, _ = self.step(batch)
        loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        return {"test_loss": loss, "pred": logits.argmax(1), "true": labels}

    def test_epoch_end(self, outputs):
        test_loss = sum([x["test_loss"] for x in outputs])
        pred = torch.cat([x["pred"] for x in outputs])
        true = torch.cat([x["true"] for x in outputs])
        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy(pred, true)
        out = {
            "test_loss": test_loss,
            "test_f_score": f_score,
            "test_accuracy": accuracy,
        }
        return {**out, "log": out}


def format_input(x):
    return f"binary classification: {x}"
