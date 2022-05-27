"""
This file contains constants for the causal evaluation part of the code.
"""
from pathlib import Path

HOME_DIR = Path.home()
PROJECT_DIR = HOME_DIR / 'CausalEvaluation'

EXPERIMENTS_DIR = PROJECT_DIR / 'experiments'
DATA_DIR = PROJECT_DIR / 'data'

SENTIMENT_ACCEPTABILITY_DOMAIN_DIR = EXPERIMENTS_DIR / 'sentiment_acceptability_domain'
