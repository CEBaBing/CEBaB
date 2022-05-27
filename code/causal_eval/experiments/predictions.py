import argparse
import os

import torch
from transformers import AutoTokenizer

from methods.causalm import BertCausalmForNonlinearSequenceClassification
from methods.utils.modeling_utils import BertForNonlinearSequenceClassification
from methods.utils.constants import BERT, ID_COL, LABEL_COL
from methods.utils.data_utils import get_cebab, task_to_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--model_architecture', type=str)
    parser.add_argument('--is_causalm', type=bool, default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--padding', type=str, default='max_length')
    parser.add_argument('--max_seq_length', type=int, default=128)
    args = parser.parse_args()

    # model
    if args.model_architecture == BERT:
        if args.is_causalm:
            model = BertCausalmForNonlinearSequenceClassification.from_pretrained(args.model_name_or_path)
        else:
            model = BertForNonlinearSequenceClassification.from_pretrained(args.model_name_or_path)

        model.to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError(f'Unsupported architecture: {args.model_architecture}')

    # data
    key, _ = task_to_keys[args.task_name]
    cebab = get_cebab(args.task_name)

    def predict_proba(x):
        encoded = tokenizer(x[key], padding=args.padding, truncation=True, return_tensors='pt')
        logits = model(**(encoded.to(args.device))).logits[0]
        probas = torch.softmax(logits, dim=-1)
        return {f'proba_class_{k}': probas[k].item() for k in range(model.num_labels)}

    with torch.no_grad():
        cebab['test'] = cebab['test'].map(predict_proba, batch_size=args.batch_size, desc='Predict proba')

    df = cebab['test'].to_pandas()
    df = df[[ID_COL] + [f'proba_class_{k}' for k in range(model.num_labels)]]
    print(df.head())

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    path = f'{args.output_dir}/test.csv'
    df.to_csv(path, index=False)
    print(f'>>> Saved predictions at {path}')


if __name__ == '__main__':
    main()
