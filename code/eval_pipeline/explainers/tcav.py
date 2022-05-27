import numpy as np
import torch
from sklearn import linear_model
from sklearn.metrics import accuracy_score

from eval_pipeline.explainers.abstract_explainer import Explainer
from eval_pipeline.utils import TREATMENTS, OVERALL_LABEL, DESCRIPTION, POSITIVE, NEGATIVE, UNKNOWN
from eval_pipeline.utils import unpack_batches


class TCAV(Explainer):
    def __init__(self, treatments=TREATMENTS, device='cpu', batch_size=64, verbose=False,
                 svm_params_dict: dict = None, num_classes=2):

        self.model = None
        self.treatments = treatments
        self.device = device
        self.batch_size = batch_size
        self.cavs = {}
        self.num_classes = num_classes
        self.verbose = verbose
        if not svm_params_dict:
            self.svm_params_dict = {
                'alpha': .01,
                'max_iter': 1000,
                'tol': 1e-3
            }

    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        self.model = classifier
        for treatment in self.treatments:
            preprocessed_dataset = self.train_preprocess(dataset, treatment)
            self.cavs[treatment] = self.learn_cav(unpack_batches(preprocessed_dataset[0]), treatment,
                                                  preprocessed_dataset[1])

    def estimate_icace(self, pairs):
        scores = []
        embeddings, intervention_types = self.test_preprocess(pairs)
        clf_head = self.model.get_classification_head()
        for idx, embedding in enumerate(embeddings):
            cav = self.cavs[intervention_types.iloc[idx]]
            grad = self.get_gradient(clf_head, torch.tensor(embedding, requires_grad=True, dtype=torch.float).to(
                self.model.device))
            classes_score = []
            for i in range(self.num_classes):
                classes_score.append(torch.tanh(torch.tensor(cav @ grad[i])).item())
            scores.append(np.array(classes_score))
        return scores

    def get_gradient(self, classifier, embedding):
        embedding.retain_grad()
        grads = []
        for k in range(self.num_classes):
            outputs = classifier(embedding)
            classifier.zero_grad()
            outputs[k].backward()
            grads.append(embedding.grad.detach().cpu().numpy())

        return grads

    def train_preprocess(self, dataset, treatment):
        # TODO throw it to data utils
        treatment_labels = dataset[f'{treatment}_aspect_majority'].map({POSITIVE or NEGATIVE: 1, UNKNOWN: 0}).dropna()
        description = dataset[DESCRIPTION][treatment_labels.index].to_list()
        overall_labels = dataset[OVERALL_LABEL][treatment_labels.index].tolist()
        return self.model.get_embeddings(description), treatment_labels.tolist(), overall_labels

    def learn_cav(self, embeddings, concept, cav_labels):
        if len(set(cav_labels)) > 2:
            raise NotImplementedError('CAVs are binary by definition')

        # learn cav
        lm = linear_model.SGDClassifier(**self.svm_params_dict)
        lm.fit(embeddings, cav_labels)
        accuracy = accuracy_score(cav_labels, lm.predict(embeddings))

        # format cav
        cav = -1 * lm.coef_[0]  # In binary classification the concept is assigned to label 0 by default, so flip coef_.
        cav = cav / np.linalg.norm(cav)  # normalize to unit vector

        # report result
        if self.verbose:
            print(f'Learned CAV for concept: {concept}')
            print(f'\t{cav[:2]}...{cav[-2:]}')
            print(f'\tAccuracy: {accuracy * 100:.1f}%')
            print()
        return cav

    def test_preprocess(self, df):
        # TODO move this function to utils because of duplicate with tcav, and set these strings to constant
        x = np.array(unpack_batches(self.model.get_embeddings(df['description_base'].tolist())))
        intervention_types = df['intervention_type']

        return x, intervention_types

