import numpy as np
import torch
import torch.nn.functional as functional
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_pipeline.explainers import Explainer
from eval_pipeline.utils import unpack_batches, POSITIVE, NEGATIVE, DESCRIPTION, OVERALL_LABEL, UNKNOWN
from .utils_concept_shap import shapley_summation, G, ConceptScoreDataset, powerset, AdaptiveClassificationHead, completeness_score


class ConceptShap(Explainer):
    """
    1) Find m concepts denoted by unit vectors that represent linear directions in the activation space,
       given by a concept discovery algorithm. In our case, we use the CAV part of TCAV (Been et. al, 2018).

    2) Given a prediction model f(x) = h( phi(x) ) and a set of concepts vectors Cs = {c1,...,cm}, we compute the
       completeness score eta_f(c1,...,cm)   (Definition 3.1). To save redundant computations, the following function
       does it for each S in Powerset(Cs).

    3) Given a set of concepts Cs = {c1,...,cm} and some completeness score eta, we compute the ConceptSHAP
       score s_i for each concept ci (Definition 4.1).
    """

    def __init__(
            self,
            concepts,
            original_model,
            num_classes,
            batch_size=64,
            device='cpu',
            verbose=False,
            svm_params_dict: dict = None,
            g_mapping_params: dict = None
    ):
        self.concepts = concepts
        self.model = original_model
        self.embedding_dim = original_model.model.config.hidden_size
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.seed = 42
        self.num_classes = num_classes
        if not svm_params_dict:
            self.svm_params_dict = {
                'alpha': .01,
                'max_iter': 1000,
                'tol': 1e-3,
            }

        if not g_mapping_params:
            self.g_mapping_params = {
                'hidden_dim': 500,
                'num_epochs': 50,
                'batch_size': 128,
                'lr': 1e-2
            }

        self.cavs = dict()

    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        for concept in self.concepts:
            if self.verbose:
                print(f'Learning CAV for concept: {concept}')
            embeddings, cav_labels, _ = self.train_preprocess(dataset, concept)
            self.cavs[concept] = self.learn_cav(embeddings, cav_labels)

    def estimate_icace(self, pairs):
        embeddings, intervention_types = self.test_preprocess(pairs)
        # classification_head = self.model.model.classifier.to(self.device)
        classification_head = self.model.get_classification_head()
        task_labels = np.array(pairs['review_majority_base'].to_list()).argmax(axis=-1)
        eta: dict = self.get_eta(embeddings, task_labels, self.cavs, classification_head)
        outputs_dict = shapley_summation(self.concepts, eta)
        outputs = [outputs_dict[intervention_type][idx].cpu().numpy() for idx, intervention_type in enumerate(intervention_types)]
        probas = [np.exp(x) / np.sum(np.exp(x), axis=0) - 0.5 for x in outputs]  # normalize back to probas
        return probas

    def get_sup_g(self, embeddings: Tensor, true_labels: Tensor, concepts_mat: Tensor, h: nn.Module):
        """
        Trains g() : R^num_concepts -> R^embedding_dim, the mapping from the concept space to the activation space.
        It's conceptually equivalent to empirically calculating sup_g.
        """
        m = concepts_mat.shape[1] if concepts_mat is not None else 0  # num concepts
        g = G(m, self.g_mapping_params['hidden_dim'], self.embedding_dim).to(self.device)  # mapping from the concept space to the activation space
        concept_scores = embeddings @ concepts_mat if concepts_mat is not None else embeddings
        true_labels.to(self.device)

        for p in h.parameters():
            p.requires_grad = False

        model = nn.Sequential(g, h)
        model = model.to(self.device)
        model.float()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.g_mapping_params['lr'])

        majority_accuracy = true_labels.float().mean().item()
        majority_accuracy = majority_accuracy if majority_accuracy > 0.5 else 1 - majority_accuracy

        dataset = ConceptScoreDataset(concept_scores, true_labels)
        dataloader = DataLoader(dataset, batch_size=self.g_mapping_params['batch_size'], shuffle=False)

        print() if self.verbose else None
        print(f'\tnumber of concepts (features): {m}    baseline accuracy: {majority_accuracy: .2f}') if self.verbose else None
        for epoch in range(self.g_mapping_params['num_epochs']):
            epoch_loss = 0.
            predictions = []
            for batch_idx, (v_c_x, y) in enumerate(dataloader):
                y = functional.one_hot(y, num_classes=self.num_classes).to(self.device).float()

                y_hat = model(v_c_x)

                loss = functional.binary_cross_entropy_with_logits(y_hat, y)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                predictions += list(torch.argmax(y_hat, dim=-1).cpu().numpy())

            if epoch % (self.g_mapping_params['num_epochs'] // 5) == 0:
                accuracy = accuracy_score(true_labels.cpu().numpy(), predictions)
                print(f'\t\tepoch: {epoch:<4} epoch_loss: ~{epoch_loss / len(dataloader):.3f}    accuracy: {accuracy:.2f}') if self.verbose else None

        print() if self.verbose else None
        for p in h.parameters():
            p.requires_grad = True

        return g

    def get_eta(self, embeddings, true_labels, cavs, h: torch.nn.Module):
        embeddings = torch.tensor(embeddings, dtype=torch.float, device=self.device)
        true_labels = torch.tensor(true_labels, device=self.device)

        eta = dict()
        h = AdaptiveClassificationHead(h)
        for key in tqdm(powerset(self.concepts), desc='Computing eta', total=pow(2, len(self.concepts))):
            concepts_mat = torch.tensor(np.array([cavs[c] for c in key])).T if key else torch.zeros(self.embedding_dim, self.embedding_dim)
            concepts_mat = concepts_mat.float().to(self.device)
            g = self.get_sup_g(embeddings, true_labels, concepts_mat, h)

            with torch.no_grad():
                eta[key] = h(g(embeddings @ concepts_mat))

        print(f'Completeness of {self.concepts}: {self.compute_completeness(embeddings, true_labels, g, h):.1f}')

        return eta

    def compute_completeness(self, embeddings: Tensor, true_labels: Tensor, g: torch.nn.Module, h: torch.nn.Module):
        with torch.no_grad():
            concepts_mat = torch.tensor(np.array([self.cavs[c] for c in self.concepts])).T.float().to(self.device)
            model_logits = h(embeddings)
            concept_logits = h(g(embeddings @ concepts_mat))
            return completeness_score(true_labels, model_logits, concept_logits, self.model.model.num_labels, device=self.device)

    def train_preprocess(self, dataset, treatment):
        aspect_labels = dataset[f'{treatment}_aspect_majority'].map({POSITIVE or NEGATIVE: 1, UNKNOWN: 0}).dropna()
        description = dataset[DESCRIPTION][aspect_labels.index].to_list()
        overall_labels = dataset[OVERALL_LABEL][aspect_labels.index].tolist()
        embeddings = unpack_batches(self.model.get_embeddings(description))
        return embeddings, aspect_labels.to_list(), overall_labels

    def test_preprocess(self, df):
        x = np.array(unpack_batches(self.model.get_embeddings(df['description_base'].tolist())))
        intervention_types = df['intervention_type']

        return x, intervention_types

    def learn_cav(self, embeddings, cav_labels):
        # check input
        if len(set(cav_labels)) > 2:
            raise NotImplementedError('Supports only binary classification.')

        # learn cav
        lm = linear_model.SGDClassifier(**self.svm_params_dict)
        lm.fit(embeddings, cav_labels)
        accuracy = accuracy_score(cav_labels, lm.predict(embeddings))

        # format cav
        cav = -1 * lm.coef_[0]  # In binary classification the concept is assigned to label 0 by default, so flip coef_.
        cav = cav / np.linalg.norm(cav)  # normalize to unit vector

        # report result
        if self.verbose:
            print(f'{cav[:2]}...{cav[-2:]}')
            print(f'Accuracy: {accuracy * 100:.1f}%')
            print()

        return cav
