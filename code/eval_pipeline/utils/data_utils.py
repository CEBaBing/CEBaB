import numpy as np
import pandas as pd


def unpack_batches(batches):
    embeddings = []
    for batch in batches:
        for embedded in batch:
            embeddings.append(embedded)
    return embeddings


def preprocess_hf_dataset(dataset, one_example_per_world=False, verbose=0, dataset_type='5-way'):
    """
    Preprocess the CEBaB dataset loaded from HuggingFace.

    Drop 'no majority' data, encode all labels as ints.
    """
    assert dataset_type in ['2-way', '3-way', '5-way']

    # only use one example per exogenous world setting if required
    if one_example_per_world:
        dataset['train'] = dataset['train_exclusive']
    else:
        dataset['train'] = dataset['train_inclusive']

    train = dataset['train'].to_pandas()
    dev = dataset['validation'].to_pandas()
    test = dataset['test'].to_pandas()

    # drop no majority reviews
    train_no_majority = train['review_majority'] == 'no majority'
    if verbose:
        percentage = 100 * sum(train_no_majority) / len(train)
        print(f'Dropping no majority reviews: {round(percentage, 4)}% of train dataset.')
    train = train[~train_no_majority]

    # encode datasets
    train = _encode_dataset(train, verbose=verbose, dataset_type=dataset_type)
    dev = _encode_dataset(dev, verbose=verbose, dataset_type=dataset_type)
    test = _encode_dataset(test, verbose=verbose, dataset_type=dataset_type)

    # fill NAs with the empty string
    aspect_columns = list(filter(lambda col: 'aspect' in col, list(train.columns)))

    train[aspect_columns] = train[aspect_columns].fillna('')
    dev[aspect_columns] = dev[aspect_columns].fillna('')
    test[aspect_columns] = test[aspect_columns].fillna('')

    return train, dev, test


def _encode_dataset(dataset, verbose=0, dataset_type='5-way'):
    """
    Encode the review and aspect columns.
    For 2-way experiments, drop neutral reviews.
    """
    # drop neutral in 2-way setting:
    if dataset_type == '2-way':
        neutral = dataset['review_majority'] == '3'
        dataset = dataset[~neutral]
        if verbose:
            print(f'Dropped {sum(neutral)} examples with a neutral label.')

    # encode dataset with the dataset_type
    encoding = None
    if dataset_type == '2-way':
        encoding = {
            "1": 0,
            "2": 0,
            "4": 1,
            "5": 1,
        }
    elif dataset_type == '3-way':
        encoding = {
            "1": 0,
            "2": 0,
            "3": 1,
            "4": 2,
            "5": 2
        }
    elif dataset_type == "5-way":
        encoding = {
            "1": 0,
            "2": 1,
            "3": 2,
            "4": 3,
            "5": 4
        }
    dataset['review_majority'] = dataset['review_majority'].apply(lambda score: encoding[score])
    return dataset


def _get_pairs_per_original(df):
    """
    For a df containing all examples related to one original,
    create and return all the possible intervention pairs.
    """
    assert len(df.original_id.unique()) == 1

    df_edit = df[~df['is_original']].reset_index(drop=True)
    if len(df_edit):
        df_original = pd.concat([df[df['is_original']]] * len(df_edit)).reset_index(drop=True)
    else:
        df_original = df[df['is_original']].reset_index(drop=True)

    assert (len(df_original) == 0) or (len(df_edit) == 0) or (len(df_edit) == len(df_original))

    # (edit, original) pairs
    edit_original_pairs = None
    original_edit_pairs = None
    if len(df_original) and len(df_edit):
        df_edit_base = df_edit.rename(columns=lambda x: x + '_base')
        df_original_counterfactual = df_original.rename(columns=lambda x: x + '_counterfactual')

        edit_original_pairs = pd.concat([df_edit_base, df_original_counterfactual], axis=1)

        # (original, edit) pairs
        df_edit_counterfactual = df_edit.rename(columns=lambda x: x + '_counterfactual')
        df_original_edit = df_original.rename(columns=lambda x: x + '_base')

        original_edit_pairs = pd.concat([df_original_edit, df_edit_counterfactual], axis=1)

    # (edit, edit) pairs
    edit_edit_pairs = None
    if len(df_edit):
        # The edits are joined based on their edit type. 
        # Actually, the 'edit_type' can also differ from the edit performed, but there is no clean way of resolving this.
        edit_edit_pairs = df_edit.merge(df_edit, on='edit_type', how='inner', suffixes=('_base', '_counterfactual'))
        edit_edit_pairs = edit_edit_pairs[edit_edit_pairs['id_base'] != edit_edit_pairs['id_counterfactual']]
        edit_edit_pairs = edit_edit_pairs.rename(columns={'edit_type': 'edit_type_base'})
        edit_edit_pairs['edit_type_counterfactual'] = edit_edit_pairs['edit_type_base']

    # get all pairs
    pairs = pd.concat([edit_original_pairs, original_edit_pairs, edit_edit_pairs]).reset_index(drop=True)

    # annotate pairs with the intervention type and the direction (calculated from the validated labels)
    pairs = _get_intervention_type_and_direction(pairs)

    return pairs


def _drop_unsuccessful_edits(pairs, verbose=0):
    """
    Drop edits that produce no measured aspect change.
    """
    # Make sure the validated labels of the edited aspects are different.
    # We can not do this comparison based on 'edit_goal_*' because the final label might differ from the goal.
    meaningless_edits = pairs['intervention_aspect_base'] == pairs['intervention_aspect_counterfactual']
    if verbose:
        print(
            f'Dropped {sum(meaningless_edits)} pairs that produced no validated label change.'
            f' This is due to faulty edits by the workers or edits with the same edit_goal.')
    pairs = pairs[~meaningless_edits]

    return pairs


def _get_intervention_type_and_direction(pairs):
    """
    Annotate a dataframe of pairs with their invention type 
    and the validated label of that type for base and counterfactual.
    """
    # get intervention type
    pairs['intervention_type'] = np.maximum(pairs['edit_type_base'].astype(str), pairs['edit_type_counterfactual'].astype(str))

    # get base/counterfactual value of the intervention aspect
    pairs['intervention_aspect_base'] = \
        ((pairs['intervention_type'] == 'ambiance') * pairs['ambiance_aspect_majority_base']) + \
        ((pairs['intervention_type'] == 'noise') * pairs['noise_aspect_majority_base']) + \
        ((pairs['intervention_type'] == 'service') * pairs['service_aspect_majority_base']) + \
        ((pairs['intervention_type'] == 'food') * pairs['food_aspect_majority_base'])

    pairs['intervention_aspect_counterfactual'] = \
        ((pairs['intervention_type'] == 'ambiance') * pairs['ambiance_aspect_majority_counterfactual']) + \
        ((pairs['intervention_type'] == 'noise') * pairs['noise_aspect_majority_counterfactual']) + \
        ((pairs['intervention_type'] == 'service') * pairs['service_aspect_majority_counterfactual']) + \
        ((pairs['intervention_type'] == 'food') * pairs['food_aspect_majority_counterfactual'])

    return pairs


def _int_to_onehot(series, rng):
    """
    Encode a series of ints as a series of onehot vectors.
    Assumes the series of ints is contained within the range.
    """
    offset = rng[0]
    rng = max(rng) - min(rng) + 1

    def _get_onehot(x):
        zeros = np.zeros(rng)
        zeros[int(x) - offset] = 1.0
        return zeros

    return series.apply(_get_onehot)


def _pairs_to_onehot(pairs, dataset_type="5-way"):
    """
    Cast the review majority columns to onehot vectors.
    """
    rng = None
    if dataset_type == '2-way':
        rng = range(0, 2)
    elif dataset_type == '3-way':
        rng = range(0, 3)
    elif dataset_type == '5-way':
        rng = range(0, 5)
    pairs['review_majority_counterfactual'] = _int_to_onehot(pairs['review_majority_counterfactual'], rng)
    pairs['review_majority_base'] = _int_to_onehot(pairs['review_majority_base'], rng)

    return pairs


def get_intervention_pairs(df, dataset_type="5-way", verbose=0):
    """
    Given a dataframe in the CEBaB data scheme, return all intervention pairs.
    """
    assert dataset_type in ['2-way', '3-way', '5-way']

    # Drop label distribution and worker information.
    columns_to_keep = ['id', 'original_id', 'edit_id', 'is_original', 'edit_goal', 'edit_type', 'description', 'review_majority',
                       'food_aspect_majority', 'ambiance_aspect_majority', 'service_aspect_majority', 'noise_aspect_majority', 'opentable_metadata']
    columns_to_keep += [col for col in df.columns if 'prediction' in col]
    df = df[columns_to_keep]

    # get all the intervention pairs
    unique_originals = df.original_id.unique()
    to_merge = []
    for unique_id in unique_originals:
        df_slice = df[df['original_id'] == unique_id]
        if len(df_slice) > 1:
            pairs_slice = _get_pairs_per_original(df_slice)
            to_merge.append(pairs_slice)
    pairs = pd.concat(to_merge)

    # drop unsuccessful edits
    pairs = _drop_unsuccessful_edits(pairs, verbose=verbose)

    # onehot encode
    pairs = _pairs_to_onehot(pairs, dataset_type=dataset_type)

    return pairs
