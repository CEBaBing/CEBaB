import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers import GPT2PreTrainedModel, GPT2Model
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.file_utils import ModelOutput

from .configuration_gpt2 import GPT2CausalmConfig
from .supported_heads import SEQUENCE_CLASSIFICATION, HEAD_TYPES
from ..configuration_causalm import CausalmHeadConfig

__all__ = [
    'GPT2ForCausalmAdditionalPreTraining',
    'GPT2CausalmForSequenceClassification',
    'GPT2ForNonlinearSequenceClassification',
    'GPT2CausalmForNonlinearSequenceClassification'
]


@dataclass
class GPT2ForCausalmAdditionalPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    tc_logits: torch.FloatTensor = None
    cc_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class GPT2CausalmAdditionalPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.tc_heads, self.tc_heads_types = self.__init_causalm_heads(config, 'tc')
        self.cc_heads, self.cc_heads_types = self.__init_causalm_heads(config, 'cc')

    def forward(self, hidden_states, pooled_hidden_states):
        lm_head_scores = self.lm_head(hidden_states)

        tc_heads_scores = []
        for tc_head, head_type in zip(self.tc_heads.values(), self.tc_heads_types):
            if head_type == SEQUENCE_CLASSIFICATION:
                tc_heads_scores.append(tc_head(pooled_hidden_states))

        cc_heads_scores = []
        for cc_head, head_type in zip(self.cc_heads.values(), self.cc_heads_types):
            if head_type == SEQUENCE_CLASSIFICATION:
                cc_heads_scores.append(cc_head(pooled_hidden_states))

        return lm_head_scores, tc_heads_scores, cc_heads_scores

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
            if not isinstance(head_cfg, CausalmHeadConfig):
                head_cfg = CausalmHeadConfig(**head_cfg)
            if head_cfg.head_type == SEQUENCE_CLASSIFICATION:
                heads[head_cfg.head_name] = nn.Sequential(
                    nn.Dropout(head_cfg.hidden_dropout_prob),
                    nn.Linear(config.hidden_size, head_cfg.num_labels)
                )
            else:
                raise NotImplementedError()

            head_types.append(head_cfg.head_type)
        return heads, head_types


class GPT2ForCausalmAdditionalPreTraining(GPT2PreTrainedModel):
    config_class = GPT2CausalmConfig
    base_model_prefix = "gpt2_causalm"

    def __init__(self, config: GPT2CausalmConfig):
        super().__init__(config)
        self.config = config

        self.transformer = GPT2Model(config)
        self.additional_pretraining_heads = GPT2CausalmAdditionalPreTrainingHeads(config)
        self.model_parallel = False

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

        transformer_outputs = self.transformer(
            input_ids,
            # past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # mimic the function of "pooler"
        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
                self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                # logger.warning(
                #     f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                #     f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                # )

        pooled_hidden_states = hidden_states[torch.arange(batch_size, device=self.device), sequence_lengths]

        mlm_head_scores, tc_heads_scores, cc_heads_scores = self.additional_pretraining_heads(hidden_states, pooled_hidden_states)

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
            output = (mlm_head_scores, tc_heads_scores, cc_heads_scores) + transformer_outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return GPT2ForCausalmAdditionalPreTrainingOutput(
            loss=total_loss,
            mlm_logits=mlm_head_scores,
            tc_logits=tc_heads_scores,
            cc_logits=cc_heads_scores,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def get_counterfactual_weights(self):
        return self.transformer


class GPT2CausalmForSequenceClassification(GPT2PreTrainedModel):
    config_class = GPT2CausalmConfig
    base_model_prefix = "gpt2_causalm"

    def __init__(self, config: GPT2CausalmConfig, cf_model: GPT2PreTrainedModel = None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        if self.config.sequence_classifier_type not in {'task', 'cc', 'tc'}:
            raise RuntimeError(f'Illegal sequence_classifier_type {self.config.sequence_classifier_type}')
        self.classifier_type = self.config.sequence_classifier_type

        self.transformer = GPT2Model(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

        # assign the adversarially trained model to be the underlying task model and freeze its weights
        if not cf_model:
            warnings.warn('Trying to initialize a CausaLM task model without providing counterfactual weights.')
        else:
            self.transformer = cf_model

        for param in self.transformer.parameters():
            param.requires_grad = False

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

        transformer_outputs = self.transformer(
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
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
                self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                # logger.warning(
                #     f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                #     f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                # )

        pooled_logits = logits[torch.arange(batch_size, device=self.device), sequence_lengths]

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
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


# Non Linear Models
class GPT2NonlinearClassificationHead(nn.Module):
    """Head for sentence-level classification tasks. Identical to RobertaClassificationHead."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        classifier_dropout = (
            config.summary_first_dropout  # 0.1
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.n_embd, config.num_labels)

    def forward(self, features, **kwargs):
        x = features  # features is the pooled [CLS] token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class GPT2ForNonlinearSequenceClassification(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = GPT2NonlinearClassificationHead(config)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
                self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                # logger.warning(
                #     f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                #     f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                # )

        pooled_logits = logits[torch.arange(batch_size, device=self.device), sequence_lengths]

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
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class GPT2CausalmForNonlinearSequenceClassification(GPT2CausalmForSequenceClassification):
    def __init__(self, config, cf_model=None):
        super(GPT2CausalmForNonlinearSequenceClassification, self).__init__(config, cf_model)
        self.classifier = GPT2NonlinearClassificationHead(config)
