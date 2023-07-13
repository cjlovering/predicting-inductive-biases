import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics as metrics
from transformers import AdamW


class BaseClassifier(pl.LightningModule):
    """The structure for the various modules is largely the same.
    For some models the step, forward, and configure_optimizers functions
    will have to be over-ridden.
    """

    def __init__(self, num_classes=2):
        super(BaseClassifier, self).__init__()
        self.val_acc = metrics.Accuracy(num_classes=num_classes)
        self.test_acc = metrics.Accuracy(num_classes=num_classes)
        self.val_f1 = metrics.F1Score(num_classes=num_classes)
        self.test_f1 = metrics.F1Score(num_classes=num_classes)

    def step(self, batch):
        device = self.encoder.device
        texts, labels = batch
        tokenized = self.tokenizer.batch_encode_plus(
            texts, add_special_tokens=True, return_tensors="pt", padding=True
        )["input_ids"].to(device)
        encoded = self.encoder(
            tokenized,
            labels=labels.to(device),
            return_dict=True,
        )
        return encoded.logits, encoded.loss

    def forward(self, batch):
        logits, _ = self.step(batch)
        return logits

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        _, loss = self.step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        _, labels = batch
        logits, loss = self.step(batch)
        self.val_acc(logits, labels)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=False)
        self.val_f1(logits, labels)
        self.log("val_f1", self.val_f1, on_step=True, on_epoch=False)
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        _, labels = batch
        logits, _ = self.step(batch)
        loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        self.test_acc(logits, labels)
        # self.log("test_acc", self.test_acc, on_step=True, on_epoch=False)
        self.test_f1(logits, labels)
        # self.log("test_f1", self.test_f1, on_step=True, on_epoch=False)
        # self.log("test_loss", loss, on_step=True, on_epoch=False)
        self.log_dict({
            'test_loss': loss,
            'test_f1': self.test_f1,
            'test_acc': self.test_acc
        })
        # return {"test_loss": loss}
