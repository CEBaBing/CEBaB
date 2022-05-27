from itertools import chain
from typing import Dict, Union, Any

import datasets
import numpy as np
import torch
from datasets import load_metric
from sklearn.metrics import accuracy_score
from torch import no_grad
from torch import softmax, mean
from torch import abs as pt_abs
from torch import sum as pt_sum
from torch.utils.data import Dataset, SequentialSampler, DataLoader
from transformers import is_datasets_available


def calc_accuracy_from_logits(outputs, true_labels, model):
    logits = outputs.cpu().numpy()
    scores = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    predictions = [{"label": item.argmax(), "score": item.max().item()} for item in scores]
    accuracy = accuracy_score(y_true=true_labels.cpu(), y_pred=[item['label'] for item in predictions])
    return accuracy, predictions


class CausalMetrics:
    def __init__(self, data_collator, batch_size=32, drop_last=True, device='cuda'):
        self.data_collator = data_collator
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = device

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)

        return inputs

    @staticmethod
    def _remove_unused_columns(dataset):
        return dataset.remove_columns(list(set(dataset.column_names)
                                           .difference({'input_ids', 'attention_mask', 'token_type_ids'})))

    def _get_dataloader(self, dataset) -> DataLoader:
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset)

        if isinstance(dataset, torch.utils.data.dataset.IterableDataset):
            return DataLoader(dataset, batch_size=self.batch_size, drop_last=self.drop_last, collate_fn=self.data_collator)

        sampler = SequentialSampler(dataset)

        return DataLoader(dataset, sampler=sampler, batch_size=self.batch_size, drop_last=self.drop_last, collate_fn=self.data_collator)

    def compute_class_expectation(self, model, dataset, cls):
        model.eval()
        dataloader = self._get_dataloader(dataset)

        with no_grad():
            expectation = 0.
            for inputs in dataloader:
                inputs = self._prepare_inputs(inputs)
                outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], token_type_ids=inputs['token_type_ids'])
                cls_probabilities = softmax(outputs.logits, dim=-1)
                expectation += mean(cls_probabilities, dim=0)[cls].item()
            return expectation / len(dataloader)

    def conexp(self, model, dataset, tc_indicator_name, cls=0) -> float:
        # TODO generalize to non-binary concept indicator
        dataset_0 = dataset.filter(lambda example: example[tc_indicator_name] == 0)
        dataset_1 = dataset.filter(lambda example: example[tc_indicator_name] == 1)

        e_0 = self.compute_class_expectation(model, dataset_0, cls)
        e_1 = self.compute_class_expectation(model, dataset_1, cls)
        conexp = abs(e_1 - e_0)

        return conexp

    def treate(self, model_o, model_cf, dataset, cls=0) -> float:
        model_o.eval()
        model_cf.eval()
        dataloader = self._get_dataloader(dataset)

        with no_grad():
            treate = 0.
            for inputs in dataloader:
                inputs = self._prepare_inputs(inputs)

                outputs_o = model_o(**inputs)
                outputs_cf = model_cf(**inputs)

                cls_probabilities_o = softmax(outputs_o.logits, dim=-1)
                cls_probabilities_cf = softmax(outputs_cf.logits, dim=-1)

                treate += pt_abs(cls_probabilities_o - cls_probabilities_cf)[:, cls].mean().item()

            treate /= len(dataloader)

        return treate

    def ate(self, model, dataset_f, dataset_cf, cls=0) -> float:
        model.eval()
        dataloader_f = self._get_dataloader(dataset_f)
        dataloader_cf = self._get_dataloader(dataset_cf)

        assert len(dataset_f) == len(dataset_cf)
        assert len(dataloader_f) == len(dataloader_cf)

        with no_grad():
            ate = 0.
            for inputs_f, inputs_cf in zip(dataloader_f, dataloader_cf):
                inputs_f = self._prepare_inputs(inputs_f)
                inputs_cf = self._prepare_inputs(inputs_cf)

                outputs_f = model(**inputs_f)
                outputs_cf = model(**inputs_cf)

                cls_probabilities_f = softmax(outputs_f.logits, dim=-1)
                cls_probabilities_cf = softmax(outputs_cf.logits, dim=-1)

                ate += pt_abs(cls_probabilities_f - cls_probabilities_cf)[:, cls].mean().item()

            ate /= len(dataloader_f)

        return ate


class EvalMetrics:
    def __init__(self, labels_list):
        self.f1 = load_metric('f1')
        self.seqeval = load_metric('seqeval')
        self.labels_list = labels_list

    def compute_token_classification_f1(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.seqeval.compute(predictions=true_predictions, references=true_labels)
        return {"f1": results["overall_f1"]}

    def compute_sequence_classification_f1(self, p):
        preds, labels = p
        preds = np.argmax(preds, axis=1)
        return self.f1.compute(predictions=preds, references=labels)
