import torch
import torch.nn as nn

import pytorch_lightning as pl

from transformers import (
    AdamW,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_cosine_schedule_with_warmup,
)

import pytorch_lightning.metrics.functional as metrics


class RobertaClassifier(pl.LightningModule):
    def __init__(self, model, num_steps, num_classes=2):
        super(RobertaClassifier, self).__init__()
        self.tokenizer = RobertaTokenizer.from_pretrained(model)
        self.encoder = RobertaForSequenceClassification.from_pretrained(model)
        self.num_steps = num_steps

    def step(self, batch):
        texts, labels = batch
        tokenized = self.tokenizer.batch_encode_plus(
            texts, add_special_tokens=True, return_tensors="pt", padding=True
        )["input_ids"]
        loss, logits = self.encoder(tokenized, labels=labels)
        return logits, loss

    def forward(self, batch):
        logits, _ = self.step(batch)
        return logits

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        _, loss = self.step(batch)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        training_loss = sum([x["loss"] for x in outputs])
        return {"train_loss": training_loss, "log": {"train_loss": training_loss}}

    def validation_step(self, batch, batch_idx):
        _, labels = batch
        logits, loss = self.step(batch)
        return {"val_loss": loss, "pred": logits.argmax(1), "true": labels}

    def validation_epoch_end(self, outputs):
        val_loss = sum([x["val_loss"] for x in outputs])
        pred = torch.cat([x["pred"] for x in outputs])
        true = torch.cat([x["true"] for x in outputs])
        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy(pred, true)
        out = {"val_loss": val_loss, "val_f_score": f_score, "val_accuracy": accuracy}
        return {**out, "log": out}

    def test_step(self, batch, batch_idx):
        _, labels = batch
        logits, loss = self.step(batch)
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
