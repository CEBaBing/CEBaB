import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from eval_pipeline.models.abstract_model import Model


class simple_lr(Model):
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('logreg', LogisticRegression())
        ])

    def __str__(self):
        return 'LR'

    def preprocess(self, df, shuffle=False):
        if shuffle:
            df_shuffled = df.copy().sample(frac=1.0).reset_index(drop=True)
        else:
            df_shuffled = df.copy()
        x = df_shuffled['description']
        y = df_shuffled['review_majority'].replace('no majority', '0').astype(int)

        return np.array(x), np.array(y)

    def fit(self, dataset):
        x, y = self.preprocess(dataset, shuffle=True)
        self.model.fit(x, y)

    def predict_proba(self, dataset):
        x, y = self.preprocess(dataset)
        probas = self.model.predict_proba(x).round(decimals=4)
        predictions = np.argmax(probas, axis=1)

        clf_report = classification_report(y, predictions, output_dict=True)

        return probas, clf_report
