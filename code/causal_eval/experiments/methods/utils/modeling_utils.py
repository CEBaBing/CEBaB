import torch
from torch import nn
from transformers import BertForSequenceClassification


class BertNonlinearClassificationHead(nn.Module):
    """Head for sentence-level classification tasks. Identical to RobertaClassificationHead."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features  # features is the pooled [CLS] token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BertForNonlinearSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
        super(BertForNonlinearSequenceClassification, self).__init__(config)
        self.classifier = BertNonlinearClassificationHead(config)
        self.post_init()
