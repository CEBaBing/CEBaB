import datasets

from .constants import OPENTABLE_BINARY, OPENTABLE_TERNARY, OPENTABLE_5_WAY, LABEL_COL, AUTH_TOKEN_PATH, NO_MAJORITY, CAUSALM_TC_LABEL_COL, \
    CAUSALM_CC_LABEL_COL, NEGATIVE, UNKNOWN, POSITIVE

task_to_keys = {
    "opentable": ("text", None),
    OPENTABLE_BINARY: ("description", None),
    OPENTABLE_TERNARY: ("description", None),
    OPENTABLE_5_WAY: ("description", None),
}


def get_auth_token(auth_token_path=AUTH_TOKEN_PATH):
    with open(auth_token_path) as f:
        return f.read()


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
    cebab = datasets.load_dataset('CEBaB/CEBaB', use_auth_token=get_auth_token())

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

    # TODO make this more elegant
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


def preprocess_cebab_for_causalm(raw_datasets, tc_labels_col, cc_labels_col=None):
    # tc labels
    raw_datasets = raw_datasets.rename_column(tc_labels_col, CAUSALM_TC_LABEL_COL)
    raw_datasets = raw_datasets.filter(lambda example: example[CAUSALM_TC_LABEL_COL] is not None)
    raw_datasets = raw_datasets.filter(lambda example: example[CAUSALM_TC_LABEL_COL] != '')
    raw_datasets = raw_datasets.filter(lambda example: example[CAUSALM_TC_LABEL_COL] != NO_MAJORITY)
    raw_datasets = raw_datasets.map(process_aspect, fn_kwargs=dict(tc=True))

    # cc labels
    if cc_labels_col:
        raw_datasets = raw_datasets.rename_column(cc_labels_col, CAUSALM_CC_LABEL_COL)
        raw_datasets = raw_datasets.filter(lambda example: example[CAUSALM_CC_LABEL_COL] is not None)
        raw_datasets = raw_datasets.filter(lambda example: example[CAUSALM_CC_LABEL_COL] != '')
        raw_datasets = raw_datasets.filter(lambda example: example[CAUSALM_CC_LABEL_COL] != NO_MAJORITY)
        raw_datasets = raw_datasets.map(process_aspect, fn_kwargs=dict(tc=False))

    return raw_datasets


if __name__ == '__main__':
    hf_cebab = get_cebab(OPENTABLE_TERNARY)
    print()
