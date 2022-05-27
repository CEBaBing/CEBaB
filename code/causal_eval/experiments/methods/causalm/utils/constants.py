from os import path
from .main_utils import count_num_cpu_gpu
from torch import device
from torch.cuda import is_available

CAUSALM_DIR = path.dirname(path.realpath(__file__))  # This must be set to the path which specifies where the CausaLM project resides

NUM_CPU = count_num_cpu_gpu()[0]

RANDOM_SEED = 42

BERT_MODEL_CHECKPOINT = 'bert-base-uncased'  # TODO change to cased

DEVICE = device('cuda') if is_available() else device('cpu')
