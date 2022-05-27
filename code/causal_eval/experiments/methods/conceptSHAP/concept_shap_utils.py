import itertools
from itertools import chain, combinations

import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


def powerset(iterable):
    """Util function to compute the powerset of an iterable.

    Example:
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def update_eta(eta):
    updated = dict()
    for k in eta.keys():
        for p in itertools.permutations(k):
            updated[p] = eta[k]
    return updated


def completeness_score(y_true, model_preds, concept_preds, n_concepts):
    """Assumes this is already argmax-ed"""
    concept_correct = torch.sum(torch.eq(y_true, concept_preds))  # numerator
    model_correct = torch.sum(torch.eq(y_true, model_preds))  # denominator
    random_accuracy = 1 / n_concepts  # a_r
    return torch.div(concept_correct - random_accuracy, model_correct - random_accuracy)  # final formula


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


def get_sup_g(args, embeddings: torch.Tensor, true_labels: torch.Tensor, concepts_mat: torch.Tensor, h: nn.Module):
    """
    Trains g() : R^num_concepts -> R^embedding_dim, the mapping from the concept space to the activation space.
    It's conceptually equivalent to empirically calculating sup_g.
    """
    m = concepts_mat.shape[1] if concepts_mat is not None else 0  # num concepts
    g = G(m, args.g_hidden_dim, args.embedding_dim).to(args.device)  # mapping from the concept space to the activation space
    concept_scores = embeddings @ concepts_mat if concepts_mat is not None else embeddings
    true_labels.to(args.device)

    for p in h.parameters():
        p.requires_grad = False

    model = nn.Sequential(g, h)
    model = model.to(args.device)
    model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.g_learning_rate)

    majority_accuracy = true_labels.float().mean().item()
    majority_accuracy = majority_accuracy if majority_accuracy > 0.5 else 1 - majority_accuracy

    dataset = ConceptScoreDataset(concept_scores, true_labels)
    dataloader = DataLoader(dataset, batch_size=args.g_batch_size, shuffle=False)

    print() if args.verbose else None
    print(f'\tnumber of concepts (features): {m}    baseline accuracy: {majority_accuracy: .2f}') if args.verbose else None
    for epoch in range(args.g_num_epochs):
        epoch_loss = 0.
        predictions = []
        for batch_idx, (v_c_x, y) in enumerate(dataloader):
            y = F.one_hot(y, num_classes=2).to(args.device).float()

            y_hat = model(v_c_x)

            loss = F.binary_cross_entropy_with_logits(y_hat, y)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predictions += list(torch.argmax(y_hat, dim=-1).cpu().numpy())

        if epoch % (args.g_num_epochs // 5) == 0:
            accuracy = accuracy_score(true_labels.cpu().numpy(), predictions)
            print(f'\t\tepoch: {epoch:<4} epoch_loss: ~{epoch_loss / len(dataloader):.3f}    accuracy: {accuracy:.2f}') if args.verbose else None

    print() if args.verbose else None
    for p in h.parameters():
        p.requires_grad = True

    return g


def to_pandas(hf_dataset):
    df = hf_dataset.to_pandas()
    embedding_dim = len(df.iloc[0]['embedding'])
    df.index = df['idx']
    df[list(range(embedding_dim))] = pd.DataFrame(df.embedding.tolist(), index=df.index)
    df = df.drop(columns=['idx', 'embedding'])
    return df
