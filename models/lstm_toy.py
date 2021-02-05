import os

import torch

import pytorch_lightning as pl
import torch.nn as nn

import pytorch_lightning.metrics.functional as metrics
from torch.optim import Adam


from . import head


class LstmToyClassifier(pl.LightningModule):
    def __init__(
        self, model, vocab_size=50_000, num_classes=2, hidden_size: int = 300,
    ):
        super(LstmToyClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.classifier = head.ClassificationHead(hidden_size, num_classes)

    def forward(self, batch):
        texts, _ = batch
        tokens = self.tokenize(texts)
        embeddings = self.embedding(tokens)
        _, (ht, _) = self.lstm(embeddings)
        logits = self.classifier(ht[-1])
        return logits

    def tokenize(self, texts):
        # All sequences are the same length in the toy setting for now,
        # so we don't need to pad, but just in case we change that in the future...
        return torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([int(w) for w in t.split()]) for t in texts],
            batch_first=True,
        )

    def training_step(self, batch, batch_idx):
        _, labels = batch
        logits = self.forward(batch)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        training_loss = sum([x["loss"] for x in outputs])
        return {"train_loss": training_loss, "log": {"train_loss": training_loss}}

    def validation_step(self, batch, batch_idx):
        _, labels = batch
        logits = self.forward(batch)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"val_loss": loss, "pred": logits.argmax(1), "true": labels}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x["val_loss"] for x in outputs]).sum()
        pred = torch.cat([x["pred"] for x in outputs])
        true = torch.cat([x["true"] for x in outputs])
        f_score = metrics.f1_score(pred, true)
        accuracy = metrics.accuracy(pred, true)
        out = {
            "val_loss": val_loss,
            "val_f_score": f_score,
            "val_accuracy": accuracy,
        }
        return {**out, "log": out}

    def test_step(self, batch, batch_idx):
        _, labels = batch
        logits = self.forward(batch)
        loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        return {"test_loss": loss, "pred": logits.argmax(1), "true": labels}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x["test_loss"] for x in outputs]).sum()
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

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return [optimizer]


def create_emb_layer(weights_matrix, trainable):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({"weight": torch.tensor(weights_matrix)})
    emb_layer.weight.requires_grad = trainable
    return emb_layer
