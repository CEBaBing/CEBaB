from itertools import chain, combinations, permutations
from math import factorial
from typing import List

import torch
from torch import Tensor
from torch import nn
from torch.utils.data import Dataset
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead


class G(nn.Module):
    """
    Implements sup_g from the completeness definition.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(G, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False) if input_dim > 0 else None
        self.bias1 = nn.Parameter(torch.randn(hidden_dim))
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if len(x.size()) < 2:
            raise RuntimeError('Must have batch dimension.')
        batch_size = x.size(0)

        if self.linear1:
            out = self.linear1(x) + self.bias1
        else:
            out = self.bias1.repeat(batch_size, 1)
        out = self.linear2(out)
        return out


class ConceptScoreDataset(Dataset):
    def __init__(self, X, y):
        super(ConceptScoreDataset).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def update_eta(eta):
    updated = dict()
    for k in eta.keys():
        for p in permutations(k):
            updated[p] = eta[k]
    return updated


def powerset(iterable):
    """Util function to compute the powerset of an iterable.

    Example:
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def shapley_summation(Cs: List[str], eta: dict):
    """
    Given a list of concepts Cs and a completeness scores dictionary n for each of the
    elements in P(Cs), computes the ConceptSHAP score s_i for each concept c_i in Cs.

    For more information please review Definition 4.1 in the ConceptSHAP paper.

    Params:
    -------
        concepts: A list containing the names of the concepts.
        eta: A dict containing the completeness scores eta(S) for each S in P(Cs).

    Usage example:
    --------------
        Input:
        ---
        concepts = ['food', 'ambiance', 'service', 'noise']
        eta = {
            (): 1.24,
            ('food',): 1.31,
            ('ambiance',): 1.29,
            ...
            ('food', 'ambiance', 'service', 'noise'): 1.76
        }

        Output:
        ---
        scores = {
            'food': 0.15,
            'ambiance': 0.12,
            'service': 0.09,
            'noise': 0.05
        }
    """
    eta = update_eta(eta)
    m = len(Cs)
    scores = dict()
    for c_i in Cs:
        s_i = torch.zeros_like(eta[tuple()])
        for S in [set(x) for x in powerset(set(Cs).difference({c_i}))]:  # S \subseteq Cs - c_i
            coef = (factorial(m - len(S) - 1) * factorial(len(S))) / factorial(m)
            s_i += coef * (eta[tuple(S.union({c_i}))] - eta[tuple(S)])
        scores[c_i] = s_i
    return scores


def completeness_score(y_true: Tensor, model_logits: Tensor, concept_logits: Tensor, num_classes: int, device='cpu') -> float:
    # get predictions
    y_true = y_true.to(device)
    concept_preds = concept_logits.argmax(dim=-1).to(device)
    model_preds = model_logits.argmax(dim=-1).to(device)

    # calculate completeness
    concept_correct = torch.sum(torch.eq(y_true, concept_preds))  # numerator
    model_correct = torch.sum(torch.eq(y_true, model_preds))  # denominator
    random_accuracy = 1 / num_classes  # a_r
    completeness = torch.div(concept_correct - random_accuracy, model_correct - random_accuracy)  # final formula

    return completeness.item()


class AdaptiveClassificationHead(nn.Module):
    """
    Dummy module that identifies which classification head is used
    and make sures all the tensor dimension are correct.
    """

    def __init__(self, classification_head):
        super(AdaptiveClassificationHead, self).__init__()
        self.h = classification_head

    def forward(self, x):
        if isinstance(self.h, RobertaClassificationHead):
            x = x.unsqueeze(1)
        x = self.h(x)
        return x.squeeze()
