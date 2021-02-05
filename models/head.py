import torch.nn as nn


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks.
    
    This is the same architecture used by/for Roberta, as well as our previous work.
    (Note, in the past we used RELU over TANH.)
    """

    def __init__(self, hidden_size, num_classes, hidden_dropout_prob=0.5):
        super().__init__()
        self.classifier_head = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, features):
        return self.classifier_head(features)
