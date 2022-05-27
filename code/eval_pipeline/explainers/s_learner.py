import numpy as np
from tqdm import tqdm
from math import ceil
from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer
import torch

from eval_pipeline.explainers.abstract_explainer import Explainer
from eval_pipeline.customized_models.bert import BertForNonlinearSequenceClassification
from eval_pipeline.explainers.explainer_utils import dataset_aspects_to_onehot 


class SLearner(Explainer):
    def __init__(self, absa_model_path, device='cpu', batch_size = 64):
        self.model = Pipeline(
            [
                ("lr", LogisticRegression())
            ]
        )

        self.device = device
        self.batch_size = batch_size

        self.aspects = ['food', 'service', 'noise', 'ambiance']

        self.absa_model = BertForNonlinearSequenceClassification.from_pretrained(absa_model_path).to(self.device)
        if 'CEBaB/' in absa_model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(absa_model_path.split('/')[1].split('.')[0])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(absa_model_path)

        self.x_train = None
        self.y_train = None
        self.x_dev = None


    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        # store representations and corresponding model predictions
        self.x_train = dataset_aspects_to_onehot(dataset)
        self.y_train = np.argmax(classifier_predictions, axis=1)
        
        # train a LR model
        self.model = self.model.fit(self.x_train, self.y_train)
        

    @staticmethod
    def _get_representations_after_interventions(pairs):
        """
        Simulate interventions in the explainable representation space.
        """
        pairs_after_intervention = pairs.copy()
        for aspect in ['food', 'service', 'ambiance', 'noise']:
            pairs_after_intervention[f'{aspect}_aspect_majority_base'] = ((pairs_after_intervention['intervention_type'] == aspect) *
                                                                          pairs_after_intervention['intervention_aspect_counterfactual']) + (
                                                                                     (pairs_after_intervention['intervention_type'] != aspect) *
                                                                                     pairs_after_intervention[f'{aspect}_aspect_majority_base'])

        return dataset_aspects_to_onehot(pairs_after_intervention.rename(columns=lambda col: col.replace('_base', '')))


    def _get_pairs_with_predicted_aspect_labels(self, pairs):
        """
        Use the ABSA model to predict the labels of the aspects that are not intervened upon.
        """
        self.absa_model.to(self.device)
        self.absa_model.eval()

        # create absa inputs
        text = pairs['description_base'].to_list()

        n_batches = ceil(len(text)/self.batch_size)
        predictions = defaultdict(list)

        # for every aspect
        for aspect in self.aspects:
            # create ABSA inputs
            absa_input = self.tokenizer(text, [aspect]*len(text), return_tensors='pt', padding=True, truncation=True).to(self.device)

            # for every batch
            for i in range(n_batches):

                # pass to model
                absa_input_batch = {k: v[i*self.batch_size: (i+1)*self.batch_size] for k,v in absa_input.items()}
                absa_output_batch = self.absa_model(**absa_input_batch)

                predictions[aspect].append(absa_output_batch.logits.detach().cpu())

        # stack batches and get argmax
        predictions = {k: np.argmax(torch.concat(v).numpy(), axis=1) for k,v in predictions.items()}

        # LIME uses different encodings than absa model
        absa_to_lime_encodings = {
            0:'Negative',
            1:'Positive',
            2:'unknown'
        }

        encoder = np.vectorize(absa_to_lime_encodings.get)
        predictions = {k: encoder(v) for k,v in predictions.items()}

        # overwrite current labels
        for aspect in self.aspects:
            pairs[f'{aspect}_aspect_majority_base'] = predictions[aspect]

        return pairs 


    def estimate_icace(self, pairs):
        # use predicted aspect labels for the base examples, instead of the ground truth labels
        pairs_predicted = self._get_pairs_with_predicted_aspect_labels(pairs.copy())

        # get the aspect encodings for the dev set
        self.x_dev = dataset_aspects_to_onehot(pairs_predicted.rename(columns=lambda col: col.replace('_base', '')))

        # apply the correct intervention in the representation space
        counterfactual_dev = self._get_representations_after_interventions(pairs_predicted)

        # get estimates
        counterfactual_probas = self.model.predict_proba(counterfactual_dev)
        factual_probas = self.model.predict_proba(self.x_dev)
        
        estimates = counterfactual_probas - factual_probas
        return list(estimates)
