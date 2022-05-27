import os
import shutil
from math import ceil

import numpy as np
import torch
from transformers import AutoTokenizer

from causal_eval.experiments.methods.causalm import BertCausalmConfig
from eval_pipeline.customized_models.bert import BertForNonlinearSequenceClassification
from .modeling import (
    BertCausalmForNonlinearSequenceClassification,
    LSTMCausalmForNonlinearSequenceClassification,
    GPT2CausalmForNonlinearSequenceClassification,
    RobertaCausalmForSequenceClassification, GPT2CausalmConfig, RobertaCausalmConfig, LSTMCausalmConfig,
)
from .. import Explainer
from ...utils import BERT, GPT2, ROBERTA, LSTM


class CausaLM(Explainer):
    def __init__(self, factual_model_path, ambiance_model_path, food_model_path, noise_model_path, service_model_path, device='cpu', batch_size=64,
                 empty_cache_after_run=False, fasttext_embeddings_path=None):
        self.device = device
        self.batch_size = batch_size
        self.factual_model_path = factual_model_path
        self.counterfactual_model_paths = {
            'food': food_model_path,
            'service': service_model_path,
            'ambiance': ambiance_model_path,
            'noise': noise_model_path
        }

        if 'CEBaB/' in ambiance_model_path:
            model_architecture = factual_model_path.split('/')[1].split('.')[0]
            if model_architecture == LSTM:  # LSTM model uses bert-base uncased tokenizer
                tokenizer_name = BERT
                if fasttext_embeddings_path:
                    self.fasttext_embeddings_path = fasttext_embeddings_path  # for LSTM
                else:
                    raise RuntimeError('If architecture is LSTM, you must provide a fasttext_embeddings_path')
            else:
                tokenizer_name = model_architecture
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

            if model_architecture == GPT2:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(ambiance_model_path)

        self.seed = factual_model_path.split('_')[-1]
        self.empty_cache_after_run = empty_cache_after_run

    def preprocess(self, df):
        x = self.tokenizer(df['description_base'].to_list(), padding=True, truncation=True, return_tensors='pt')
        intervention_types = df['intervention_type']

        return x, intervention_types

    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        # Assume CausaLM has been trained offline
        pass

    def estimate_icace(self, pairs):
        # preprocess
        x, intervention_types = self.preprocess(pairs)

        # load the factual model
        # TODO: work with variable model architectures
        f_model = self.get_model(self.factual_model_path).to(self.device)

        # for every type of causaLM model
        aspect_to_probas = {}
        aspect_to_mask = {}
        for aspect, aspect_model_path in self.counterfactual_model_paths.items():
            probas_aspect = []

            # load the counterfactual model and tokenizer
            # TODO: work with variable model architectures
            cf_model = self.get_model(aspect_model_path).to(self.device)
            cf_model.eval()

            # get subset of data corresponding with this intervention type
            mask = (intervention_types == aspect).to_numpy()
            x_aspect = {k: v[mask] for k, v in x.items()}

            # run the model in batches
            for i in range(ceil(len(x_aspect['input_ids']) / self.batch_size)):
                # get the difference between the factual and counterfactual model
                x_batch = {k: v[i * self.batch_size:(i + 1) * self.batch_size].to(self.device) for k, v in x_aspect.items()}
                cf_model_probas = torch.nn.functional.softmax(cf_model(**x_batch).logits.cpu(), dim=-1).detach()
                f_model_probas = torch.nn.functional.softmax(f_model(**x_batch).logits.cpu(), dim=-1).detach()
                probas_aspect.append(cf_model_probas - f_model_probas)

            # save predictions
            aspect_to_probas[aspect] = probas_aspect
            aspect_to_mask[aspect] = mask

        # merge the predictions
        num_labels = aspect_to_probas['food'][0].shape[1]
        probas = np.zeros((len(x['input_ids']), num_labels))
        for aspect, probas_per_aspect in aspect_to_probas.items():
            probas[aspect_to_mask[aspect]] = torch.concat(probas_per_aspect)
        probas = np.round(probas, decimals=4)

        # if required, clean the HF cache 
        if self.empty_cache_after_run:
            home = os.path.expanduser('~')
            hf_cache = os.path.join(home, '.cache', 'huggingface', 'transformers')
            print(f'Deleting HuggingFace cache at {hf_cache}.')
            shutil.rmtree(hf_cache, ignore_errors=True)

        return list(probas)

    def get_model(self, pretrained_path):
        model_architecture = pretrained_path.split('/')[-1].split('.')[0]

        config_kwargs = dict()
        # determine model architecture and instantiate config and model accordingly
        if model_architecture == BERT:
            config_class = BertCausalmConfig
            model_class = BertCausalmForNonlinearSequenceClassification

        elif model_architecture == GPT2:
            config_class = GPT2CausalmConfig
            model_class = GPT2CausalmForNonlinearSequenceClassification

        elif model_architecture == ROBERTA:
            config_class = RobertaCausalmConfig
            model_class = RobertaCausalmForSequenceClassification  # RobertaClassificationHead is non-linear

        elif model_architecture == LSTM:
            config_kwargs['fasttext_embeddings_path'] = self.fasttext_embeddings_path
            config_class = LSTMCausalmConfig
            model_class = LSTMCausalmForNonlinearSequenceClassification

        else:
            raise RuntimeError(f'Unsupported architecture "{model_architecture}"')

        config = config_class.from_pretrained(pretrained_path, **config_kwargs, use_auth_token=True)
        if model_architecture == GPT2:
            config.pad_token_id = self.tokenizer.pad_token_id
        model = model_class.from_pretrained(pretrained_path, config=config, use_auth_token=True)

        return model
