PROJECT_NAME = 'CausaLM'

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

# wandb
WANDB_PROJECT = PROJECT_NAME

# models stuff
ROBERTA_VOCAB_SIZE = 50265

# CEBaB
CEBAB = 'CEBaB'
