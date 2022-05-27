import argparse
import pickle

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root_dir', type=str,
                        default='/home/eldar.a/OpenTable/causal_eval/experiments/outputs_concept_shap_scores')
    parser.add_argument('--treated_concepts', nargs='+',
                        default=['rating_food', 'rating_ambiance', 'rating_service', 'rating_noise'])
    parser.add_argument('--confounding_concepts', nargs='+',
                        default=['None', 'cuisine', 'price_tier', 'region', 'rating_food',
                                 'rating_ambiance', 'rating_service', 'rating_noise'])
    parser.add_argument('--directions', nargs='+',
                        default=['Positive_to_Negative', 'Positive_to_unknown', 'Negative_to_Positive',
                                 'Negative_to_unknown', 'unknown_to_Positive', 'unknown_to_Negative']
                        )
    parser.add_argument('--seeds', nargs='+', default=list(range(42, 47)))
    args = parser.parse_args()

    results_df = aggregate_results(**vars(args))

    results_df.to_csv(f'{args.results_root_dir}/concept_shap_results.csv', index=False)


def aggregate_results(results_root_dir, treated_concepts, confounding_concepts, directions, seeds):
    df = []
    for tc in treated_concepts:
        for cc in confounding_concepts:
            if cc == tc:
                continue
            for d in directions:
                for seed in seeds:
                    with open(f'{results_root_dir}/fold_scores_{tc}___{cc}_{seed}.pkl', 'rb') as f:
                        fold_scores_dict = pickle.load(f)
                        effect = fold_scores_dict[f'test_{d}_f'][tc.split('_')[-1]]
                        df.append((tc, cc, d, seed, effect))
    df = pd.DataFrame.from_records(df, columns=['tc', 'cc', 'direction', 'seed', 'effect'])
    return df


if __name__ == '__main__':
    main()
