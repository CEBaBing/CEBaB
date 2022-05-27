import numpy as np

from .abstract_explainer import Explainer

class RandomExplainer(Explainer):
    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        pass

    @staticmethod
    def _get_random_probability_vectors(L,N):
        p = np.random.uniform(size=(L, N))
        p = p / np.repeat(np.expand_dims(np.linalg.norm(p,axis=1, ord=1), -1), N, axis=-1)
        return p

    def estimate_icace(self, pairs):
        # get the number of classes to predict for
        N = pairs['review_majority_base'].iloc[0].shape[0]
        L = len(pairs)
        
        # generate random (counter)factual predictions
        factual_predictions = self._get_random_probability_vectors(L,N)
        counterfactual_predictions = self._get_random_probability_vectors(L,N)

        # difference the predictions
        estimates = counterfactual_predictions - factual_predictions
        return list(estimates)