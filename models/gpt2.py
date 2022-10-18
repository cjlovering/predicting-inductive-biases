from transformers import (
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
)
from .base import BaseClassifier


class GPT2Classifier(BaseClassifier):
    def __init__(self, model, num_steps, num_classes=2):
        super(GPT2Classifier, self).__init__(num_classes)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.encoder = GPT2ForSequenceClassification.from_pretrained(model)
        self.encoder.config.pad_token_id = self.tokenizer.eos_token_id
        self.num_steps = num_steps
