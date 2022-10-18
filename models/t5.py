import torch.nn as nn
from transformers import (
    AdamW,
    T5Tokenizer,
    T5Model,
    get_linear_schedule_with_warmup,
)
from .base import BaseClassifier

class T5Classifier(BaseClassifier):
    def __init__(self, model, num_steps, num_classes=2):
        super(T5Classifier, self).__init__(num_classes)
        hidden_size = {"t5-small": 512, "t5-base": 768, "t5-large": 1024,}[model]
        self.model = T5Model.from_pretrained(model)
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        self.num_steps = num_steps
        self.classifier = nn.Linear(hidden_size, num_classes)

    def step(self, batch):
        device = self.device
        texts, labels = batch
        texts = [format_input(t) for t in texts]
        input_ids = self.tokenizer.batch_encode_plus(
            texts, padding=True, return_tensors="pt", max_length=64
        ).to(device)
        outputs = self.model.encoder(
            **{k:v.to(device) for k,v in input_ids.items()},
            return_dict=False,
        )
        last_hidden_states = outputs[0][:, -1, :]
        logits = self.classifier(last_hidden_states)
        loss = nn.functional.cross_entropy(logits, labels.to(device))
        return logits, loss

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

def format_input(x):
    return f"binary classification: {x}"
