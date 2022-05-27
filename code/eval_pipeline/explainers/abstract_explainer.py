from abc import ABC, abstractmethod

import numpy as np


class Explainer(ABC):
    def __str__(self):
        return type(self).__name__

    @abstractmethod
    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        pass

    @abstractmethod
    def estimate_icace(self, pairs):
        pass


class ZeroExplainer(Explainer):
    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        pass

    def estimate_icace(self, pairs):
        # get the number of classes to predict for
        N = pairs['review_majority_base'].iloc[0].shape[0]
        L = len(pairs)

        return list(np.zeros((L, N)))
