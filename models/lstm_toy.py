import torch
import torch.nn as nn
from torch.optim import Adam

from . import head
from .base import BaseClassifier


class LstmToyClassifier(BaseClassifier):
    def __init__(
        self,
        model,
        vocab_size=50_000,
        num_classes=2,
        hidden_size: int = 300,
    ):
        super(LstmToyClassifier, self).__init__(num_classes)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.classifier = head.ClassificationHead(hidden_size, num_classes)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        return [optimizer]

    def step(self, batch):
        device = self.device
        texts, labels = batch
        tokens = self.tokenize(texts).to(device)
        embeddings = self.embedding(tokens)
        _, (ht, _) = self.lstm(embeddings)
        logits = self.classifier(ht[-1])
        loss = nn.functional.cross_entropy(logits, labels.to(device))
        return logits, loss

    def tokenize(self, texts):
        # All sequences are the same length in the toy setting for now,
        # so we don't need to pad, but just in case we change that in the future...
        return torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([int(w) for w in t.split()]) for t in texts],
            batch_first=True,
        )
