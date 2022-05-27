import argparse
import json
import logging
import os
import pickle
from math import factorial
from typing import List

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import set_seed, AutoModelForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from methods.utils.constants import TASK_NAMES
from methods.utils.data_utils import get_cebab
from methods.utils.methods_utils import get_embeddings, learn_cav, create_cav_labels
from methods.conceptSHAP.concept_shap_utils import powerset, update_eta, completeness_score, get_sup_g

logger = logging.getLogger(__name__)


def get_completeness_score_dict(args, embeddings, true_labels, cavs, h: torch.nn.Module):
    embeddings = torch.tensor(embeddings, dtype=torch.float, device=args.device)
    true_labels = torch.tensor(true_labels, device=args.device)

    class H(nn.Module):
        def __init__(self, h):
            super(H, self).__init__()
            self.h = h

        def forward(self, x):
            if isinstance(self.h, RobertaClassificationHead):
                x = x.unsqueeze(1)
            x = self.h(x)
            return x.squeeze()

    result = dict()
    for key in tqdm(powerset(args.concepts), desc='Completeness score dict', total=pow(2, len(args.concepts))):
        h = H(h)
        concepts_mat = torch.tensor(np.array([cavs[c] for c in key]), dtype=torch.float, device=args.device).T if key else None
        g = get_sup_g(args, embeddings, true_labels, concepts_mat, h)

        with torch.no_grad():
            model_preds = torch.tensor([torch.argmax(h(x_i.to(args.device))) for x_i in embeddings]).to(args.device)
            concept_scores = embeddings @ concepts_mat if concepts_mat is not None else embeddings
            concept_preds = torch.tensor([torch.argmax(h(g(v_c_x.unsqueeze(0).to(args.device)))) for v_c_x in concept_scores]).to(args.device)
            n_concepts = len(args.concepts)

            result[key] = completeness_score(true_labels, model_preds, concept_preds, n_concepts).item()

    return result


def shapley_scores(Cs: List[str], eta: dict):
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
        s_i = 0
        for S in [set(x) for x in powerset(set(Cs).difference({c_i}))]:  # S \subseteq Cs - c_i
            coef = (factorial(m - len(S) - 1) * factorial(len(S))) / factorial(m)
            s_i += coef * (eta[tuple(S.union({c_i}))] - eta[tuple(S)])
        scores[c_i] = s_i
    return scores


def concept_shap(args, h, embeddings_dfs_dict, cav_labels_dfs_dict):
    """
    The ConceptSHAP algorithm. To operate the algorithm, you only need to call this in your main.
    """
    # 1) Find m concepts denoted by unit vectors that represent linear directions in the activation space,
    #    given by a concept discovery algorithm. In our case, we use the CAV part of TCAV (Been et. al, 2018).
    #    We are using TRAINING data.
    cavs = dict()
    for c in args.concepts:
        cavs_df = cav_labels_dfs_dict[c].join(embeddings_dfs_dict['train']).dropna()
        cav_train_embeddings = cavs_df[list(range(args.embedding_dim))].to_numpy()
        cav_labels = cavs_df['cav_label'].to_numpy()
        cavs[c] = learn_cav(args, c, cav_train_embeddings, cav_labels)

    fold_scores_dict = dict()
    for fold in embeddings_dfs_dict:
        if fold == 'train':
            continue
        # 2) Given a prediction model f(x) = h( phi(x) ) and a set of concepts vectors Cs = {c1,...,cm}, we compute the
        #    completeness score eta_f(c1,...,cm)   (Definition 3.1). To save redundant computations, the following function
        #    does it for each S in Powerset(Cs).
        #    We are using TESTING data.
        test_embeddings = embeddings_dfs_dict[fold][list(range(args.embedding_dim))].to_numpy()
        test_task_labels = embeddings_dfs_dict[fold]['label'].to_numpy()
        eta_dict = get_completeness_score_dict(args, test_embeddings, test_task_labels, cavs, h)

        # 3) Given a set of concepts Cs = {c1,...,cm} and some completeness score eta, we compute the ConceptSHAP
        #    score s_i for each concept ci (Definition 4.1).
        scores = shapley_scores(args.concepts, eta_dict)
        fold_scores_dict[fold] = scores

    return fold_scores_dict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--concepts', nargs='+', default=['food', 'ambiance', 'service', 'noise'],
                        help='The list of concepts to compute the SHAP score for.')
    parser.add_argument('--task_name', type=str, help=f'Either of {TASK_NAMES}')
    parser.add_argument('--model_path', type=str)  # train, all

    # output paths
    parser.add_argument('--cavs_output_dir', type=str, default=None, help='If provided, saves the learned CAVs to disk.')
    parser.add_argument('--embeddings_output_dir', type=str, default=None, help='If provided, saves the model embeddings to disk.')
    parser.add_argument('--concept_shap_output_dir', type=str, default=None, help='If provided, saves the ConceptSHAP scores to disk.')

    # technicals
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--embedding_dim', type=int, default=768, help='Default embedding dim.')
    parser.add_argument('--batch_size', type=int, default=64, help='Default embedding dim.')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cpu/cuda. Default: cuda.')
    parser.add_argument('--verbose', type=bool, default=False, help='Weather or not to print logs.')

    # hyper-parameters: linear classifier (CAVs)
    parser.add_argument('--svm_alpha', type=float, default=.01, help='The alpha for the linear classifier.')
    parser.add_argument('--svm_max_iter', type=int, default=1000, help='Max iterations for the linear classifier.')
    parser.add_argument('--svm_tol', type=float, default=1e-3, help='Error tolerance for the linear classifier.')

    # hyper-parameters: g (completeness score)
    parser.add_argument('--g_hidden_dim', type=int, default=500)
    parser.add_argument('--g_learning_rate', type=float, default=1e-2)
    parser.add_argument('--g_num_epochs', type=int, default=50)
    parser.add_argument('--g_batch_size', type=int, default=128)

    arguments = parser.parse_args()

    set_seed(arguments.seed)

    # data
    cebab = get_cebab(arguments.task_name)
    train_df = cebab['train'].to_pandas()
    cav_labels_dfs_dict = create_cav_labels(arguments, train_df)

    # algorithm prerequisites
    embeddings_dfs_dict = get_embeddings(arguments, cebab, arguments.model_path)
    h = AutoModelForSequenceClassification.from_pretrained(arguments.model_path).classifier.to(arguments.device)

    # ConceptSHAP
    fold_scores_dict = concept_shap(arguments, h, embeddings_dfs_dict, cav_labels_dfs_dict)

    # report results
    if arguments.verbose:
        print()
        print(f'ConceptSHAP scores:')
        for fold, concept_score_dict in fold_scores_dict.items():
            if fold.split('_')[-1] == 'cf':
                continue
            print(f'fold: {fold}')
            for concept, score in concept_score_dict.items():
                print(f'\t{concept:<10} {score:.3f}')
            print()

    # save results
    if arguments.concept_shap_output_dir:
        if not os.path.isdir(arguments.concept_shap_output_dir):
            os.makedirs(arguments.concept_shap_output_dir)
        with open(f'{arguments.concept_shap_output_dir}/concept_shap_scores.pkl', 'wb') as f:
            pickle.dump(fold_scores_dict, f)
        with open(f'{arguments.concept_shap_output_dir}/arguments.json', 'w') as f:
            json.dump(vars(arguments), f)


if __name__ == '__main__':
    main()
