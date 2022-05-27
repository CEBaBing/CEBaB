import argparse

from utils.data_utils import get_cebab
from utils.methods_utils import get_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--embeddings_output_dir', type=str, help='The root path for the output.')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size for embedding.')
    parser.add_argument('--verbose', default=True, type=bool)
    args = parser.parse_args()

    raw_datasets = get_cebab(args.task_name)
    get_embeddings(args, raw_datasets, args.model_name_or_path)


if __name__ == '__main__':
    main()
