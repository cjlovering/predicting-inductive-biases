from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
)

from .base import BaseClassifier

class RobertaClassifier(BaseClassifier):
    def __init__(self, model, num_steps, num_classes=2):
        super(RobertaClassifier, self).__init__(num_classes)
        self.tokenizer = RobertaTokenizer.from_pretrained(model)
        self.encoder = RobertaForSequenceClassification.from_pretrained(model)
        self.num_steps = num_steps