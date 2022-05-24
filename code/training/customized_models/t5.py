import torch
from torch import nn
from transformers import T5ForConditionalGeneration


class T5NonlinearLMHead(nn.Module):
    """Head for sentence-level classification tasks. Identical to RobertaClassificationHead."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model)
        classifier_dropout = (
            config.dropout_rate if config.dropout_rate is not None else 0.1
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, features, **kwargs):
        x = features  # features is the pooled [CLS] token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class T5ForNonlinearConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super(T5ForNonlinearConditionalGeneration, self).__init__(config)
        self.lm_head = T5NonlinearLMHead(config)
        super().post_init()

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.out_proj = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head.out_proj