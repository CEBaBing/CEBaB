import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from transformers import AutoModel, AutoTokenizer

from .constants import NO_MAJORITY, UNKNOWN, ID_COL
from .data_utils import task_to_keys


def get_embeddings(args, raw_datasets, pretrained_path):
    # load cached embeddings, if exist
    if args.embeddings_output_dir and os.path.isdir(args.embeddings_output_dir):
        print(f'>>> Loading cached embeddings from: {args.embeddings_output_dir}')
        embedded_dfs_dict = dict()
        for fold in raw_datasets:
            df = pd.read_csv(f'{args.embeddings_output_dir}/{fold}.csv')
            df = df.rename(columns={str(i): i for i in range(args.embedding_dim)})
            df.index = df[ID_COL]
            embedded_dfs_dict[fold] = df
        return embedded_dfs_dict

    # load model and tokenizer
    model = AutoModel.from_pretrained(pretrained_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    # process data
    def process_func(x):
        key, _ = task_to_keys[args.task_name]
        encoded = tokenizer(x[key], padding='max_length', truncation=True, return_tensors='pt')
        embedded = model(**(encoded.to(args.device)))['pooler_output'][0].cpu().numpy()
        return {'embedding': embedded}

    with torch.no_grad():
        embedded_datasets = raw_datasets.map(process_func, batch_size=args.batch_size, desc='Embedding')

    embedded_dfs_dict = {fold: to_pandas(hf_dataset) for fold, hf_dataset in embedded_datasets.items()}

    # report result
    if args.verbose:
        print(f'Created embeddings, train head:')
        print(embedded_dfs_dict['train'].head(3))

    if args.embeddings_output_dir:
        if not os.path.isdir(args.embeddings_output_dir):
            os.makedirs(args.embeddings_output_dir)
        for fold, df in embedded_dfs_dict.items():
            df.to_csv(f'{args.embeddings_output_dir}/{fold}.csv')
        print(f'>>> Saved embeddings cache to: {args.embeddings_output_dir}')

    return embedded_dfs_dict


def to_pandas(hf_dataset):
    df = hf_dataset.to_pandas()
    embedding_dim = len(df.iloc[0]['embedding'])
    df.index = df[ID_COL]
    df[list(range(embedding_dim))] = pd.DataFrame(df.embedding.tolist(), index=df.index)
    df = df.drop(columns=['embedding'])
    return df


def create_cav_labels(args, df):
    df.index = df.id
    df = df.rename(columns={f'{c}_aspect_majority': f'{c}_cav_label' for c in args.concepts})
    df = df.dropna(subset=[f'{c}_cav_label' for c in args.concepts])
    for c in args.concepts:
        df = df[df[f'{c}_cav_label'] != NO_MAJORITY]
        df[f'{c}_cav_label'] = df[f'{c}_cav_label'].apply(lambda x: 0 if x == UNKNOWN else 1)
    dfs_dict = {c: df[[f'{c}_cav_label']] for c in args.concepts}
    dfs_dict = {c: df.rename(columns={f'{c}_cav_label': 'cav_label'}) for c, df in dfs_dict.items()}
    return dfs_dict


def learn_cav(args, concept, embeddings, cav_labels):
    # check input
    if len(set(cav_labels)) > 2:
        raise NotImplementedError('Supports only binary classification.')

    # learn cav
    lm = linear_model.SGDClassifier(alpha=args.svm_alpha, max_iter=args.svm_max_iter, tol=args.svm_tol)
    lm.fit(embeddings, cav_labels)
    accuracy = accuracy_score(cav_labels, lm.predict(embeddings))

    # format cav
    cav = -1 * lm.coef_[0]  # In binary classification the concept is assigned to label 0 by default, so flip coef_.
    cav = cav / np.linalg.norm(cav)  # normalize to unit vector

    # report result
    if args.verbose:
        print(f'Learned CAV for concept: {concept}')
        print(f'\t{cav[:2]}...{cav[-2:]}')
        print(f'\tAccuracy: {accuracy * 100:.1f}%')
        print()

    if args.cavs_output_dir:
        save_dict = {
            'concept': concept,
            'cav': cav,
            'accuracy': accuracy,
            'model_path': args.model_path
        }
        if not os.path.isdir(args.cavs_output_dir):
            os.makedirs(args.cavs_output_dir)
        with open(f'{args.cavs_output_dir}/cav_{concept}.pkl', 'wb') as f:
            pickle.dump(save_dict, f)

    return cav
