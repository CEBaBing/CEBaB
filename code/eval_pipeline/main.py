import argparse
import os
import shutil

import datasets
import numpy as np
import pandas as pd

from eval_pipeline.explainers import CONEXP, CausaLM, INLP, ConceptShap, TCAV, ZeroExplainer, RandomExplainer, SLearner
from eval_pipeline.models import BERTForCEBaB, RoBERTaForCEBaB, GPT2ForCEBaB, LSTMForCEBaB
from eval_pipeline.pipeline import run_pipelines
# TODO: get rid of these seed maps or describe somewhere how they work
from eval_pipeline.utils import (
    OPENTABLE_BINARY,
    OPENTABLE_TERNARY,
    OPENTABLE_5_WAY,
    BERT,
    GPT2,
    ROBERTA,
    LSTM,
    SEEDS_ELDAR2ZEN,
    preprocess_hf_dataset,
    save_output,
    average_over_seeds, SEEDS_ELDAR, TREATMENTS, CEBAB
)


def get_caces_for_ultimate_results_table(args, cebab):
    # TODO we need to get rid of this function and compute these more elegantly
    train, dev, test = preprocess_hf_dataset(cebab, one_example_per_world=True, verbose=1,
                                             dataset_type=args.dataset_type)

    # init model and explainer, assuming all are based on the same architecture
    models = [BERTForCEBaB(
        f'CEBaB/{args.model_architecture}.CEBaB.sa.{args.str_num_classes}.exclusive.seed_{SEEDS_ELDAR2ZEN[s]}',
        device=args.device)
              for s in args.seeds]
    explainers = [ZeroExplainer()] * len(models)

    # run pipeline
    df = run_pipelines(models, explainers, train, dev, dataset_type=args.dataset_type)[-1]

    relevant_directions = [('Negative', 'Positive'), ('Negative', 'unknown'), ('unknown', 'Positive')]
    relevant_rows = [idx for idx in df.index if (idx[1], idx[2]) in relevant_directions]
    relevant_cols = [col for col in df.columns if col[-1] == 'ICaCE']
    df = df.loc[relevant_rows, relevant_cols]
    for c in df.columns:
        df[c] = df[c].apply(lambda x: x[1])
    df['avg'] = df.apply(np.mean, axis=1)
    df['std'] = df.apply(np.std, axis=1)
    caces, caces_std = df['avg'].to_numpy().reshape(4, 3), df['std'].to_numpy().reshape(4, 3)

    if args.output_dir:
        filename = f'caces__{args.task_name}__bert-base-uncased.csv'
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        pd.DataFrame(caces,
                     columns=['neg to pos', 'neg to unk', 'unk to pos'],
                     index=['ambiance', 'food', 'noise', 'service']).to_csv(f'{args.output_dir}/{filename}')


def get_explainers(seed, args, model):
    causalm = CausaLM(
        factual_model_path=f'CEBaB/{args.model_architecture}.{CEBAB}.causalm.None__None.{args.str_num_classes}.exclusive.seed_{seed}',
        ambiance_model_path=f'CEBaB/{args.model_architecture}.{CEBAB}.causalm.ambiance__food.{args.str_num_classes}.exclusive.seed_{seed}',
        food_model_path=f'CEBaB/{args.model_architecture}.{CEBAB}.causalm.food__service.{args.str_num_classes}.exclusive.seed_{seed}',
        noise_model_path=f'CEBaB/{args.model_architecture}.{CEBAB}.causalm.noise__food.{args.str_num_classes}.exclusive.seed_{seed}',
        service_model_path=f'CEBaB/{args.model_architecture}.{CEBAB}.causalm.service__food.{args.str_num_classes}.exclusive.seed_{seed}',
        empty_cache_after_run=True,
        device=args.device,
        batch_size=args.batch_size,
        fasttext_embeddings_path=args.fasttext_embeddings_path  # for LSTM
    )

    # slearner always uses bert-base-uncased to predict the aspect-level labels
    slearner = SLearner(
        f'CEBaB/bert-base-uncased.CEBaB.absa.exclusive.seed_{SEEDS_ELDAR2ZEN[int(seed)]}',
        device=args.device,
        batch_size=args.batch_size
    )

    explainers = [
        RandomExplainer(),
        CONEXP(),
        slearner,
        causalm,
        TCAV(treatments=TREATMENTS, device=args.device, batch_size=args.batch_size, num_classes=args.num_classes),
        INLP(treatments=TREATMENTS, device=args.device, batch_size=args.batch_size),
        ConceptShap(concepts=TREATMENTS, original_model=model, device=args.device, verbose=args.verbose,
                    batch_size=args.batch_size, num_classes=args.num_classes)
    ]
    return explainers


def get_model(seed, args):
    path = f'CEBaB/{args.model_architecture}.CEBaB.sa.{args.str_num_classes}.exclusive.seed_{SEEDS_ELDAR2ZEN[int(seed)]}'
    if args.model_architecture == BERT:
        return BERTForCEBaB(path, device=args.device)
    elif args.model_architecture == ROBERTA:
        return RoBERTaForCEBaB(path, device=args.device)
    elif args.model_architecture == LSTM:
        return LSTMForCEBaB(path, device=args.device)
    elif args.model_architecture == GPT2:
        return GPT2ForCEBaB(path, device=args.device)


def main():
    # TODO: add explanations of these arguments or examples

    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default=OPENTABLE_BINARY)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--seeds', nargs='+', default=SEEDS_ELDAR)
    parser.add_argument('--model_architecture', type=str, default=BERT)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--eval_split', type=str, default='dev')
    parser.add_argument('--flush_cache', type=bool, default=False)
    parser.add_argument('--fasttext_embeddings_path', type=str, default='./eval_pipeline/explainers/causalm/utils/lstm_embeddings.bin')
    args = parser.parse_args()

    # data
    cebab = datasets.load_dataset('CEBaB/CEBaB', use_auth_token=True)
    if args.task_name == OPENTABLE_BINARY:
        args.num_classes = 2
    elif args.task_name == OPENTABLE_TERNARY:
        args.num_classes = 3
    elif args.task_name == OPENTABLE_5_WAY:
        args.num_classes = 5
    else:
        raise ValueError(f'Unsupported task \"{args.task_name}\"')
    args.str_num_classes = f'{args.num_classes}-class'
    args.dataset_type = f'{args.num_classes}-way'
    train, dev, test = preprocess_hf_dataset(cebab, one_example_per_world=True, verbose=1,
                                             dataset_type=args.dataset_type)

    # for every seed
    pipeline_outputs = []
    for seed in args.seeds:
        # TODO: support multiple models
        model = get_model(seed, args)

        explainers = get_explainers(seed, args, model)
        # TODO: these are shallow model copies! If one explainer manipulates a model without copying, this could give bugs for other methods!
        models = [model] * len(explainers)

        eval_dataset = dev if args.eval_split == 'dev' else test
        pipeline_output = run_pipelines(models, explainers, train, eval_dataset, dataset_type=args.dataset_type,
                                        shorten_model_name=True)
        pipeline_outputs.append(pipeline_output)

        if args.flush_cache:
            home = os.path.expanduser('~')
            hf_cache = os.path.join(home, '.cache', 'huggingface', 'transformers')
            print(f'Deleting HuggingFace cache at {hf_cache}.')
            shutil.rmtree(hf_cache, ignore_errors=True)

    # average over the seeds
    pipeline_outputs_averaged = average_over_seeds(pipeline_outputs)

    # save output
    if args.output_dir:
        filename_suffix = f'{args.task_name}__{args.model_architecture}__{args.eval_split}'
        save_output(os.path.join(args.output_dir, f'final__{filename_suffix}'), filename_suffix,
                    *pipeline_outputs_averaged)


if __name__ == '__main__':
    main()
