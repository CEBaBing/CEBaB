from pathlib import Path
from .consts import PROJECT_NAME

USER_HOME = Path.home()
PROJECT_DIR = USER_HOME / PROJECT_NAME
OUTPUTS_CAUSALM = PROJECT_DIR / 'outputs_causalm'
OUTPUTS_CEBAB = PROJECT_DIR / 'outputs_cebab'

# hf transformers
TRANSFORMERS_CACHE = PROJECT_DIR / 'hf_cache'
