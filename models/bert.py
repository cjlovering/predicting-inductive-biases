from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
)
from .base import BaseClassifier

class BertClassifier(BaseClassifier):
    def __init__(self, model, num_steps, num_classes=2):
        super(BertClassifier, self).__init__(num_classes)
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.encoder = BertForSequenceClassification.from_pretrained(model)
        self.num_steps = num_steps