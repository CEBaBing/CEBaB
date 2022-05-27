from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead

from ..utils.modeling_utils import BertNonlinearClassificationHead
from .configuration_causalm import BertCausalmConfig, CausalmHeadConfig
from .utils import SEQUENCE_CLASSIFICATION, TOKEN_CLASSIFICATION, HEAD_TYPES

__all__ = [
    'BertForCausalmAdditionalPreTraining',
    'BertCausalmForSequenceClassification',
    'BertCausalmForNonlinearSequenceClassification',
    'BertCausalmForTokenClassification'
]


@dataclass
class BertForCausalmAdditionalPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    tc_logits: torch.FloatTensor = None
    cc_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertCausalmAdditionalPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlm_head = BertLMPredictionHead(config)
        self.tc_heads, self.tc_heads_types = self.__init_causalm_heads(config, 'tc')
        self.cc_heads, self.cc_heads_types = self.__init_causalm_heads(config, 'cc')

    def forward(self, sequence_output, pooled_output):
        mlm_head_scores = self.mlm_head(sequence_output)

        tc_heads_scores = []
        for tc_head, head_type in zip(self.tc_heads.values(), self.tc_heads_types):
            if head_type == SEQUENCE_CLASSIFICATION:
                tc_heads_scores.append(tc_head(pooled_output))
            elif head_type == TOKEN_CLASSIFICATION:
                tc_heads_scores.append(tc_head(sequence_output))

        cc_heads_scores = []
        for cc_head, head_type in zip(self.cc_heads.values(), self.cc_heads_types):
            if head_type == SEQUENCE_CLASSIFICATION:
                cc_heads_scores.append(cc_head(pooled_output))
            elif head_type == TOKEN_CLASSIFICATION:
                cc_heads_scores.append(cc_head(sequence_output))

        return mlm_head_scores, tc_heads_scores, cc_heads_scores

    @staticmethod
    def __init_causalm_heads(config, mode):
        heads = nn.ModuleDict()
        head_types = []

        if mode == 'tc':
            heads_cfg = config.tc_heads_cfg if config.tc_heads_cfg else []
        elif mode == 'cc':
            heads_cfg = config.cc_heads_cfg if config.cc_heads_cfg else []
        else:
            raise RuntimeError(f'Illegal mode: "{mode}". Can be either "tc" or "cc".')

        for head_cfg in heads_cfg:
            if head_cfg.head_type == SEQUENCE_CLASSIFICATION or head_cfg.head_type == TOKEN_CLASSIFICATION:
                heads[head_cfg.head_name] = nn.Sequential(
                    nn.Dropout(head_cfg.hidden_dropout_prob),
                    nn.Linear(config.hidden_size, head_cfg.num_labels)
                )
            else:
                raise NotImplementedError()

            head_types.append(head_cfg.head_type)
        return heads, head_types


class BertForCausalmAdditionalPreTraining(BertPreTrainedModel):
    config_class = BertCausalmConfig
    base_model_prefix = "bert_causalm"

    def __init__(self, config: BertCausalmConfig):
        super().__init__(config)
        self.config = config

        self.bert = BertModel(config)
        self.additional_pretraining_heads = BertCausalmAdditionalPreTrainingHeads(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mlm_labels=None,
            tc_labels=None,
            cc_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        mlm_head_scores, tc_heads_scores, cc_heads_scores = self.additional_pretraining_heads(sequence_output, pooled_output)

        # TODO add token classification version
        for head in self.config.tc_heads_cfg + self.config.cc_heads_cfg:
            if head.head_type not in HEAD_TYPES:
                raise NotImplementedError()

        total_loss = None
        if mlm_labels is not None or tc_labels is not None or cc_labels is not None:
            loss_fct = CrossEntropyLoss()

            # LM loss
            if mlm_labels is not None:
                total_loss = loss_fct(mlm_head_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))
            else:
                raise NotImplementedError()

            # Treated concepts loss, note the minus
            if tc_labels is not None:
                for tc_head_score, tc_head_cfg in zip(tc_heads_scores, self.config.tc_heads_cfg):
                    total_loss -= self.config.tc_lambda * loss_fct(tc_head_score.view(-1, tc_head_cfg.num_labels), tc_labels.view(-1))

            # Control concepts loss
            if cc_labels is not None:
                for cc_head_score, cc_head_cfg in zip(cc_heads_scores, self.config.cc_heads_cfg):
                    total_loss += loss_fct(cc_head_score.view(-1, cc_head_cfg.num_labels), cc_labels.view(-1))

        if not return_dict:
            output = (mlm_head_scores, tc_heads_scores, cc_heads_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForCausalmAdditionalPreTrainingOutput(
            loss=total_loss,
            mlm_logits=mlm_head_scores,
            tc_logits=tc_heads_scores,
            cc_logits=cc_heads_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertCausalmForSequenceClassification(BertPreTrainedModel):
    config_class = BertCausalmConfig
    base_model_prefix = "bert_causalm"

    def __init__(self, config: BertCausalmConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        if self.config.sequence_classifier_type not in {'task', 'cc', 'tc'}:
            raise RuntimeError(f'Illegal sequence_classifier_type {self.config.sequence_classifier_type}')
        self.classifier_type = self.config.sequence_classifier_type

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            task_labels=None,
            cc_labels=None,
            tc_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        labels = None
        if self.classifier_type == 'task':
            labels = task_labels
        elif self.classifier_type == 'cc':
            labels = cc_labels
        elif self.classifier_type == 'tc':
            labels = tc_labels

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertCausalmForNonlinearSequenceClassification(BertCausalmForSequenceClassification):
    def __init__(self, config):
        super(BertCausalmForNonlinearSequenceClassification, self).__init__(config)
        self.classifier = BertNonlinearClassificationHead(config)
        self.post_init()


class BertCausalmForTokenClassification(BertPreTrainedModel):
    config_class = BertCausalmConfig
    base_model_prefix = "bert_causalm"

    def __init__(self, config: BertCausalmConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        if self.config.token_classifier_type not in {'task', 'cc', 'tc'}:
            raise RuntimeError(f'Illegal token_classifier_type {self.config.token_classifier_type}')
        self.classifier_type = self.config.token_classifier_type

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            task_labels=None,
            cc_labels=None,
            tc_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        labels = None
        if self.classifier_type == 'task':
            labels = task_labels
        elif self.classifier_type == 'cc':
            labels = cc_labels
        elif self.classifier_type == 'tc':
            labels = tc_labels

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def test_caribbean_bert():
    config = BertCausalmConfig(
        tc_heads_cfg=[CausalmHeadConfig(head_name='acceptability', head_type=SEQUENCE_CLASSIFICATION, head_params={'num_labels': 2}),
                      CausalmHeadConfig(head_name='product_specific', head_type=TOKEN_CLASSIFICATION, head_params={'num_labels': 6})],
        cc_heads_cfg=[CausalmHeadConfig(head_name='is_books', head_type=SEQUENCE_CLASSIFICATION, head_params={'num_labels': 2}),
                      CausalmHeadConfig(head_name='NER', head_type=TOKEN_CLASSIFICATION, head_params={'num_labels': 10})],
        tc_lambda=0.2,
    )
    model = BertForCausalmAdditionalPreTraining(config)
    print(model)


if __name__ == '__main__':
    test_caribbean_bert()
