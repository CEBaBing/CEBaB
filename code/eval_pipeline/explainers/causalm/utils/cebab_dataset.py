import datasets

from .consts import CAUSALM_CC_LABEL_COL, CAUSALM_TC_LABEL_COL

# data values
POSITIVE = 'Positive'
NEGATIVE = 'Negative'
UNKNOWN = 'unknown'
NO_MAJORITY = 'no majority'

# task names
OPENTABLE_BINARY = 'opentable_binary'
OPENTABLE_TERNARY = 'opentable_ternary'
OPENTABLE_5_WAY = 'opentable_5_way'

# column names
ID_COL = 'id'
LABEL_COL = 'label'
CUISINE_COL = 'cuisine'

task_to_keys = {
    OPENTABLE_BINARY: ("description", None),
    OPENTABLE_TERNARY: ("description", None),
    OPENTABLE_5_WAY: ("description", None),
}

id2label_dicts = {
    'cuisine': {0: 'american', 1: 'french', 2: 'italian', 3: 'mediterranean', 4: 'seafood'},
    'region': {0: 'midwest', 1: 'northeast', 2: 'south', 3: 'west'},
    'price_tier': {0: 'low', 1: 'med'},
    'rating_food': {0: 'quiet', 1: 'energetic'},
    'rating_ambiance': {0: 'neg', 1: 'pos'},
    'rating_service': {0: 'neg', 1: 'pos'},
    'rating_noise': {0: 'neg', 1: 'pos'}
}


def to_ternary(example):
    label = int(example['review_majority'])
    if label < 3:
        return {LABEL_COL: 0}  # negative
    elif label == 3:
        return {LABEL_COL: 1}  # neutral
    elif label > 3:
        return {LABEL_COL: 2}  # positive
    else:
        raise RuntimeError(f'Illegal label \"{label}\"')


def get_cebab(task_name):
    cebab = datasets.load_dataset('CEBaB/CEBaB', use_auth_token=True)

    # filter out all no majority examples
    cebab = cebab.filter(lambda example: example['review_majority'] != NO_MAJORITY)

    if task_name == OPENTABLE_BINARY:
        cebab = cebab.filter(lambda example: example['review_majority'] != '3')
        cebab = cebab.map(lambda example: {LABEL_COL: 1 if int(example['review_majority']) > 3 else 0})
    elif task_name == OPENTABLE_TERNARY:
        cebab = cebab.map(to_ternary)
    elif task_name == OPENTABLE_5_WAY:
        cebab = cebab.map(lambda example: {LABEL_COL: int(example['review_majority']) - 1})
    else:
        raise RuntimeError(f'Unsupported task_name \"{task_name}\".')

    cebab['train'] = cebab['train_exclusive']

    return cebab


def process_aspect(example, tc=True):
    col_name = CAUSALM_TC_LABEL_COL if tc else CAUSALM_CC_LABEL_COL
    label = example[col_name]
    if label == NEGATIVE:
        return {col_name: 0}
    elif label == UNKNOWN:
        return {col_name: 1}
    elif label == POSITIVE:
        return {col_name: 2}
    else:
        raise RuntimeError(f'Illegal label \"{label}\"')


def process_cuisine(example):
    cuisine2id = {
        'american': 0,
        'french': 1,
        'italian': 2,
        'mediterranean': 3,
        'seafood': 4,
    }
    return {CUISINE_COL: cuisine2id[example[CUISINE_COL]]}


def preprocess_cebab_for_causalm(raw_datasets, tc_labels_col, cc_labels_col=None):
    # tc labels
    raw_datasets = raw_datasets.rename_column(tc_labels_col, CAUSALM_TC_LABEL_COL)
    raw_datasets = raw_datasets.filter(lambda example: example[CAUSALM_TC_LABEL_COL] is not None)
    raw_datasets = raw_datasets.filter(lambda example: example[CAUSALM_TC_LABEL_COL] != '')
    raw_datasets = raw_datasets.filter(lambda example: example[CAUSALM_TC_LABEL_COL] != NO_MAJORITY)
    raw_datasets = raw_datasets.map(process_aspect, fn_kwargs=dict(tc=True))

    # cc labels
    if cc_labels_col:
        raw_datasets = raw_datasets.filter(lambda example: example[cc_labels_col] is not None)
        raw_datasets = raw_datasets.filter(lambda example: example[cc_labels_col] != '')
        raw_datasets = raw_datasets.filter(lambda example: example[cc_labels_col] != NO_MAJORITY)
        raw_datasets = raw_datasets.rename_column(cc_labels_col, CAUSALM_CC_LABEL_COL)
        raw_datasets = raw_datasets.map(process_aspect, fn_kwargs=dict(tc=False))

    return raw_datasets
