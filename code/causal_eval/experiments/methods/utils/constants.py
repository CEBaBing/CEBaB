from pathlib import Path

# data values
POSITIVE = 'Positive'
NEGATIVE = 'Negative'
UNKNOWN = 'unknown'
NO_MAJORITY = 'no majority'

# task names
OPENTABLE_BINARY = 'opentable_binary'
OPENTABLE_TERNARY = 'opentable_ternary'
OPENTABLE_5_WAY = 'opentable_5_way'
TASK_NAMES = [OPENTABLE_BINARY, OPENTABLE_TERNARY, OPENTABLE_5_WAY]

# columns
LABEL_COL = 'label'
ID_COL = 'id'

# paths
USER_HOME = Path.home()
PROJECT = USER_HOME / 'OpenTable'
CAUSAL_EVAL = PROJECT / 'causal_eval'
EXPERIMENTS = CAUSAL_EVAL / 'experiments'

AUTH_TOKEN_PATH = PROJECT / 'auth_token.txt'

# wandb
WANDB_PROJECT = 'CEBaB'

# model checkpoints
BERT = 'bert-base-uncased'
T5 = 't5-base'
GPT2 = 'gpt2'
ROBERTA = 'roberta-base'
LSTM = 'lstm'

MODELS = [BERT, T5, GPT2, ROBERTA, LSTM]

# causalm stuff
CAUSALM_TASK_LABEL_COL = 'task_labels'
CAUSALM_TC_LABEL_COL = 'tc_labels'
CAUSALM_CC_LABEL_COL = 'cc_labels'
