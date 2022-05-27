import numpy as np

from eval_pipeline.explainers.abstract_explainer import Explainer


class CONEXP(Explainer):
    """
    Implements the naive baseline Conditional Expectation explainer.
    """

    def __init__(self):
        self.aspects = ['food', 'service', 'ambiance', 'noise']
        self.labels = ['Positive', 'Negative', 'unknown']

        self.conditional_expectations = {
            aspect: {
                label: None for label in self.labels
            } for aspect in self.aspects
        }

    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        # create a dataset with the predictions
        dataset_with_model_predictions = dataset
        dataset_with_model_predictions['prediction'] = list(classifier_predictions)

        # calculate the conditional probabilities
        for aspect in self.aspects:
            for label in self.labels:
                filtered_dataset = dataset_with_model_predictions[dataset_with_model_predictions[f'{aspect}_aspect_majority'] == label]
                self.conditional_expectations[aspect][label] = filtered_dataset.prediction.mean()

    def estimate_icace(self, pairs):
        def conditional_difference(pair):
            intervention_type = pair['intervention_type']
            intervention_aspect_base = pair['intervention_aspect_base']
            intervention_aspect_counterfactual = pair['intervention_aspect_counterfactual']

            return self.conditional_expectations[intervention_type][intervention_aspect_counterfactual] - \
                   self.conditional_expectations[intervention_type][intervention_aspect_base]

        return pairs.apply(conditional_difference, axis=1).apply(lambda x: np.round(x, decimals=4))
