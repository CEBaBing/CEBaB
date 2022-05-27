from transformers import BertForSequenceClassification, AutoModelForSequenceClassification
import os
import pickle, json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

RESULTS_DIR = 'results'
HOME_PATH = '/home/yairgat/OpenTable/causal_eval/experiments/'
EMBEDDING_MODEL = 'trained'
EMBEDDING_PATH = os.path.join(HOME_PATH, 'outputs_embeddings', EMBEDDING_MODEL)
TRAIN_PATH = 'train.json'
VALID_PATH = 'validation.json'
SEEDS = [42]
EMBEDDING_DIM = 768
CAVS_PATH = os.path.join(HOME_PATH, 'outputs_cavs')
RESULTS_PATH = os.path.join(HOME_PATH, RESULTS_DIR)
if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)
TCAV_RESULTS_PATH = os.path.join(RESULTS_PATH, 'tcav')
if not os.path.exists(TCAV_RESULTS_PATH):
    os.mkdir(TCAV_RESULTS_PATH)
hidden_dim = 768
output_dim = 2
CLASSIFIER = torch.nn.Sequential(
    torch.nn.Linear(hidden_dim, hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dim, output_dim),
)

DIRECTIONS = ['test_Positive_to_Negative_f', 'test_Positive_to_unknown_f', 'test_Negative_to_Positive_f',
              'test_Negative_to_unknown_f',
              'test_unknown_to_Positive_f', 'test_unknown_to_Negative_f']

MODEL_PATH = 'pytorch_model.bin'
MODEL_DIR = 'task_models_nonlinear'


def load_model_by_treatment(model, seed, treatment):
    """
    Loading the model corresponds to specific seed and treatment
    :param model: consists two parts - fine tuned LM model (BERT/ROBERTA) and classifier head which trained to predict sentiment.
    :param seed: Each model was trained on a few seeds
    :param treatment: Treatment-set the model was trained on
    """

    model.classifier = CLASSIFIER
    model.load_state_dict(torch.load(
        os.path.join(HOME_PATH, MODEL_DIR, f'{treatment}__seed_{seed}', MODEL_PATH)))
    return model


def load_cav_by_treatment(rating, seed, treatment):
    """
        Loading the cav corresponds to specific seed and treatment
        :param rating: rating___treatment__concept-confounder
        :param seed: Each cav was trained on a few seeds
        :param treatment: Treatment-set the cav was trained on
    """
    with open(os.path.join(CAVS_PATH, rating, f'seed_{seed}', f'cav_{treatment}.pkl'), "rb") as f:
        f = pickle.load(f)
    return f


def get_dataset_by_path(path):
    """
         read json from the following structure {text:[], labels:[], 1:[]....embedding_size:[]}
        :param path: Path of the json file
    """
    df = pd.read_json(f'{path}.json')
    embeddings = df[[str(i) for i in range(EMBEDDING_DIM)]].to_numpy()
    text = df['text']
    labels = df['label']

    return embeddings, labels, text


def get_sentences_dict(embeddings):
    """

    """
    embeddings_dict = {}
    for idx in embeddings['0'].keys():
        sentence = torch.tensor(np.array([embeddings[k][idx] for k in list(embeddings.keys())[:-2]]),
                                requires_grad=True)
        embeddings_dict[idx] = sentence
    return embeddings_dict


def get_gradient(classifier, embedding):
    """
        Calculate the gradient of the classifier at specific point(input- embedding of sentence)
        :param classifier:
        :param embedding:
    """
    inputs = torch.autograd.Variable(embedding, requires_grad=True)
    inputs.retain_grad()
    outputs = classifier(inputs)
    logit_of_task_class = outputs[1]
    logit_of_task_class.backward()
    return inputs.grad.detach().numpy()


def TCAV_score(sensitivity_by_idx):
    """
        Get sensitivity score and return TCAV = |{input : S(input) > 0} / |all inputs||
    """
    sensitivity_np = np.array([*sensitivity_by_idx.values()])
    return np.count_nonzero(sensitivity_np > 0) / sensitivity_np.shape[0]


ratings = os.listdir(EMBEDDING_PATH)
results = {'tc': [], 'cc': [], 'direction': [], 'TCAV Score': [], 'stds': []}
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
for rating in tqdm(ratings):
    components = rating.split('___')
    treatment = components[0].split('_')[1]
    cc = components[1]
    tcav_score_per_seed = {direction: [] for direction in DIRECTIONS}
    for seed in SEEDS:
        model = load_model_by_treatment(model, seed, treatment)
        cav = load_cav_by_treatment(rating, seed, treatment)
        cav = cav['cav']
        for direction in DIRECTIONS:
            results['tc'].append(treatment), results['cc'].append(cc), results['direction'].append(direction)
            # Bert fine tuned embeddings
            ds = get_dataset_by_path(os.path.join(EMBEDDING_PATH, rating, f'seed_{seed}', direction))
            embeddings = ds[0]
            classifier = model.classifier.double()
            gradient_by_idx = {}
            sensitivity_by_idx = {}
            for idx, sentence in enumerate(embeddings):
                sentence = torch.tensor(sentence)
                grad = get_gradient(classifier, sentence)
                gradient_by_idx[idx] = grad
                sensitivity_by_idx[idx] = cav @ grad
            tcav_score_per_seed[direction].append(TCAV_score(sensitivity_by_idx))
    stds = [np.std(d) for d in tcav_score_per_seed.values()]
    tcav_means = [np.mean(d) for d in tcav_score_per_seed.values()]
    [results['stds'].append(std) for std in stds]
    [results['TCAV Score'].append(mean) for mean in tcav_means]

df = pd.DataFrame.from_dict(results)
df.to_csv(os.path.join(TCAV_RESULTS_PATH, 'TCAV_Scores.csv'))
