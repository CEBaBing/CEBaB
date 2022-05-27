import torch
from torch import nn
from transformers import RobertaForSequenceClassification


class RobertaNonlinearClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        if len(features.shape) == 3:
            x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        else:
            x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaForNonlinearSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config):
        super(RobertaForNonlinearSequenceClassification, self).__init__(config)
        self.classifier = RobertaNonlinearClassificationHead(config)
        super().post_init()
