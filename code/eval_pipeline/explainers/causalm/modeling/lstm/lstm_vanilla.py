from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput,
)


class LSTMEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id).requires_grad_(
            config.update_embeddings)
        self.word_embeddings.weight.data = torch.load(config.fasttext_embeddings_path)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class Attention(nn.Module):
    """
    Attention network
    Parameters
    ----------
    rnn_size : int
        Size of Bi-LSTM
    """

    def __init__(self, rnn_size: int) -> None:
        super(Attention, self).__init__()
        self.w = nn.Linear(rnn_size, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H : torch.Tensor (batch_size, word_pad_len, hidden_size)
            Output of Bi-LSTM
        Returns
        -------
        r : torch.Tensor (batch_size, rnn_size)
            Sentence representation
        alpha : torch.Tensor (batch_size, word_pad_len)
            Attention weights
        """
        # eq.9: M = tanh(H)
        M = self.tanh(H)  # (batch_size, word_pad_len, rnn_size)

        # eq.10: α = softmax(w^T M)
        alpha = self.w(M).squeeze(2)  # (batch_size, word_pad_len)
        alpha = self.softmax(alpha)  # (batch_size, word_pad_len)

        # eq.11: r = H
        r = H * alpha.unsqueeze(2)  # (batch_size, word_pad_len, rnn_size)
        r = r.sum(dim=1)  # (batch_size, rnn_size)

        return r, alpha


class LSTMConfig(PretrainedConfig):
    def __init__(
            self,
            fasttext_embeddings_path=None,
            vocab_size=30522,
            hidden_size=300,
            num_hidden_layers=12,
            bidirectional=True,
            update_embeddings=False,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
            use_cache=False,
            classifier_dropout=None,
            **kwargs
    ):
        super(LSTMConfig, self).__init__(pad_token_id=pad_token_id, **kwargs)

        self.fasttext_embeddings_path = fasttext_embeddings_path
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.update_embeddings = update_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.bidirectional = bidirectional
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class LSTMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LSTMConfig
    base_model_prefix = "lstm"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class LSTMModel(LSTMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = LSTMEmbeddings(config)

        # Encoder RNN
        self.rnn_size = config.hidden_size
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_hidden_layers,
            bidirectional=config.bidirectional,
            dropout=(0 if config.num_hidden_layers == 1 else config.hidden_dropout_prob),
            batch_first=True
        )
        self.tanh = nn.Tanh()

        self.attention = Attention(config.hidden_size)

    @staticmethod
    def apply_attention(rnn_output, final_hidden_state):
        """
        Apply Attention on RNN output

        Input:
            rnn_output (batch_size, seq_len, num_directions * hidden_size): tensor representing hidden state for every word in the sentence
            final_hidden_state (batch_size, num_directions * hidden_size): final hidden state of the RNN

        Returns:
            attention_output(batch_size, num_directions * hidden_size): attention output vector for the batch
        """
        hidden_state = final_hidden_state.unsqueeze(2)
        attention_scores = torch.bmm(rnn_output, hidden_state).squeeze(2)
        soft_attention_weights = functional.softmax(attention_scores, 1).unsqueeze(2)  # shape = (batch_size, seq_len, 1)
        attention_output = torch.bmm(rnn_output.permute(0, 2, 1), soft_attention_weights).squeeze(2)
        return attention_output

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        embedding_output = self.embeddings(
            input_ids=input_ids,
        )

        # pack sequences (remove word-pads, SENTENCES -> WORDS)
        packed_words = pack_padded_sequence(
            embedding_output,
            lengths=attention_mask.sum(dim=-1).tolist(),
            batch_first=True,
            enforce_sorted=False
        )  # a PackedSequence object, where 'data' is the flattened words (n_words, emb_size)

        # run through bidirectional LSTM (PyTorch automatically applies it on the PackedSequence)
        rnn_out, _ = self.lstm(packed_words)  # a PackedSequence object, where 'data' is the output of the LSTM (n_words, 2 * rnn_size)

        # unpack sequences (re-pad with 0s, WORDS -> SENTENCES)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)  # (batch_size, word_pad_len, 2 * word_rnn_size)

        # eq.8: h_i = [\overrightarrow{h}_i ⨁ \overleftarrow{h}_i ]
        # H = {h_1, h_2, ..., h_T}
        H = rnn_out[:, :, : self.rnn_size] + rnn_out[:, :, self.rnn_size:]  # (batch_size, word_pad_len, rnn_size)

        # attention module
        r, alphas = self.attention(H)  # (batch_size, rnn_size), (batch_size, word_pad_len)

        # eq.12: h* = tanh(r)
        final_out = self.tanh(r)  # (batch_size, rnn_size)

        if not return_dict:
            return (final_out, final_out) + final_out

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=final_out,
            pooler_output=final_out,
        )


class LSTMForSequenceClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.lstm = LSTMModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.lstm(
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
        )
