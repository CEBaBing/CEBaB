from scipy.linalg import null_space
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from matplotlib import pyplot as plt
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn

RESULTS_DIR = 'results_roberta_clf'
HOME_PATH = '/home/yairgat/OpenTable/causal_eval/experiments/'
EMBEDDING_MODEL = 'bert-base-uncased'
EMBEDDING_PATH = os.path.join(HOME_PATH, 'outputs_embeddings', EMBEDDING_MODEL)
TRAIN_PATH = 'train.json'
VALID_PATH = 'validation.json'
EMBEDDING_DIM = 768
RESULTS_PATH = os.path.join(HOME_PATH, RESULTS_DIR)
if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)
INLP_RESULTS_PATH = os.path.join(RESULTS_PATH, 'inlp')
if not os.path.exists(INLP_RESULTS_PATH):
    os.mkdir(INLP_RESULTS_PATH)
CLFS_PATH = os.path.join(INLP_RESULTS_PATH, 'clfs')
if not os.path.exists(CLFS_PATH):
    os.mkdir(CLFS_PATH)

INLP_EMBEDDINGS_PATH = os.path.join(INLP_RESULTS_PATH, 'inlp_embeddings')
if not os.path.exists(INLP_EMBEDDINGS_PATH):
    os.mkdir(INLP_EMBEDDINGS_PATH)

FIGS_PATH = os.path.join(INLP_RESULTS_PATH, 'CLF_FIGS')
if not os.path.exists(FIGS_PATH):
    os.mkdir(FIGS_PATH)

CLF = torch.nn.Sequential(
    torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM, bias=True),
    torch.nn.Dropout(p=0.1),
    torch.nn.Linear(EMBEDDING_DIM, 2, bias=True)
)
DIRECTIONS = ['test_Positive_to_Negative_f', 'test_Positive_to_unknown_f', 'test_Negative_to_Positive_f',
              'test_Negative_to_unknown_f',
              'test_unknown_to_Positive_f', 'test_unknown_to_Negative_f']


def get_dataset_by_path(path, is_base_set):
    """
        load the dataset(json) which located at path.
        :pram path:
        :param is_base_set: True if train or validation set, otherwise False
    """
    df = pd.read_json(path)
    embeddings = df[[str(i) for i in range(EMBEDDING_DIM)]].to_numpy()
    text = df['text']
    task_labels = df['task_labels'].to_numpy()
    if is_base_set:
        tc_labels = df['tc_labels'].to_numpy()
        return embeddings, tc_labels, task_labels, text

    return embeddings, task_labels, text


def train_clf(x_train, y_train, x_valid, y_valid, clf_model, clf_name, path_figs, path_clf):
    """
    train a classifier 'clf_model' to predict 'y_train' from 'x_train'.
    :param x_train: sentences embedding representations
    :param y_train: task concept
    :x_valid
    :y_valid
    :param clf_model: classifier
    :param clf_name:
    """
    print(f'start training {clf_name}')

    learning_rate = 0.01
    num_epochs = 250
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(clf_model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)
    train_embeddings = torch.from_numpy(x_train).float()
    train_labels = torch.from_numpy(y_train).float()
    valid_embeddings = torch.from_numpy(x_valid).float()
    valid_labels = torch.from_numpy(y_valid).float()
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(num_epochs):

        logits = clf_model(train_embeddings)
        loss = criterion(logits, train_labels.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted = torch.argmax(logits, 1)
        train_accuracy = (predicted == train_labels).sum() / len(train_labels)

        if epoch % (num_epochs // 4) == 0:
            print(f'{clf_name}- epoch: {epoch} loss: {loss} accuracy: {train_accuracy}')
            scheduler.step(train_accuracy)

        train_accuracies.append(train_accuracy)
        valid_preds = torch.argmax(clf_model(valid_embeddings), 1)
        valid_accuracy = (valid_preds == valid_labels).sum() / len(valid_labels)
        valid_accuracies.append(valid_accuracy)
    clf_model.eval()
    plt.plot(train_accuracies, color='r', label='train')
    plt.plot(valid_accuracies, color='g', label='test')
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.legend()
    plt.title(clf_name)
    plt.grid()
    plt.savefig(os.path.join(path_figs, f'{clf_name}.png'))
    plt.clf()
    print(f'{clf_name}: training accuracy: {train_accuracy}, validation accuracy: {valid_accuracy}')
    torch.save(clf_model.state_dict(), os.path.join(path_clf, f'{clf_name}.pt'))
    return clf_model


def INLP(X, y, iter=50):
    """
    Remove treated concept from X.
    :param X: Data representations X
    :param y: labels of treated concept y
    :param treated_concept:
    :param iter: INLP'S number of iteration iter.
    :return: X after INLP (X projected onto the null space of linear classifiers that predict y).
    """
    X_projected = np.array(X)
    accuracy = []
    for i in np.arange(iter):
        clf = LogisticRegression(random_state=5).fit(X_projected, y)
        w = clf.coef_
        predictions = clf.predict(X_projected)
        accuracy.append(accuracy_score(predictions, y))
        b = null_space(w)
        p_null_space = b @ b.T
        X_projected = (p_null_space @ X_projected.T).T
    return X_projected


def save_dataset_to_path(embeddings, text, tc_labels, task_labels, path=''):
    """
        create and save params as json file
        :param embeddings:
        :param text:
        :param tc_labels:
        :param task_labels:
        :param path:
    """
    df = pd.DataFrame()
    df['text'] = text
    df['tc_labels'] = tc_labels.tolist()
    df['task_labels'] = task_labels.tolist()
    df[list(range(EMBEDDING_DIM))] = pd.DataFrame(embeddings.tolist(), index=df.index)
    df.to_json(f'{path}.json')


ratings = os.listdir(EMBEDDING_PATH)
results = {'tc': [], 'cc': [], 'direction': [], 'ATE': []}
for rating in tqdm(ratings):
    components = rating.split('___')
    treatment = components[0]
    cc = components[1]
    # corresponds to seed
    data_by_treatment_path = os.path.join(EMBEDDING_PATH, rating)
    data = os.listdir(data_by_treatment_path)
    train_valid_paths = {'train': os.path.join(data_by_treatment_path, TRAIN_PATH),
                         'valid': os.path.join(data_by_treatment_path, VALID_PATH)}
    train_valid_sets = {'train': get_dataset_by_path(train_valid_paths['train'], is_base_set=True),
                        'valid': get_dataset_by_path(train_valid_paths['valid'], is_base_set=True)}

    figs_by_tc_cc_path = os.path.join(FIGS_PATH, rating)
    if not os.path.exists(figs_by_tc_cc_path):
        os.mkdir(figs_by_tc_cc_path)

    clfs_by_tc_cc_path = os.path.join(CLFS_PATH, rating)
    if not os.path.exists(clfs_by_tc_cc_path):
        os.mkdir(clfs_by_tc_cc_path)

    clf = train_clf(x_train=train_valid_sets['train'][0], y_train=train_valid_sets['train'][2],
                    x_valid=train_valid_sets['valid'][0], y_valid=train_valid_sets['valid'][2], clf_model=deepcopy(CLF),
                    clf_name=rating, path_figs=figs_by_tc_cc_path, path_clf=clfs_by_tc_cc_path)

    inlp_embeddings = INLP(train_valid_sets['train'][0], train_valid_sets['train'][1])

    clf_inlp = train_clf(x_train=inlp_embeddings, y_train=train_valid_sets['train'][2],
                         x_valid=train_valid_sets['valid'][0], y_valid=train_valid_sets['valid'][2],
                         clf_model=deepcopy(CLF),
                         clf_name=f'{rating}_INLP', path_figs=figs_by_tc_cc_path, path_clf=clfs_by_tc_cc_path)

    save_dataset_to_path(text=train_valid_sets['train'][3], tc_labels=train_valid_sets['train'][1],
                         task_labels=train_valid_sets['train'][2], embeddings=inlp_embeddings,
                         path=os.path.join(INLP_EMBEDDINGS_PATH, rating))

    for direction in tqdm(DIRECTIONS):
        direction_path = os.path.join(data_by_treatment_path, f'{direction}.json')
        direction_embeddings = get_dataset_by_path(direction_path, is_base_set=False)[0]
        with torch.no_grad():
            logits = clf(torch.tensor(direction_embeddings.astype(np.float32)))
            logits_INLP = clf_inlp(torch.tensor(direction_embeddings.astype(np.float32)))

        mean_probs = torch.softmax(logits, dim=1)[:, 1].mean()
        mean_probs_inlp = torch.softmax(logits_INLP, dim=1)[:, 1].mean()
        ate = mean_probs - mean_probs_inlp
        results['tc'].append(treatment), results['cc'].append(cc), results['direction'].append(direction)
        results['ATE'].append(float(ate))

df = pd.DataFrame.from_dict(results)
df.to_csv(os.path.join(INLP_RESULTS_PATH, 'ATE.csv'))
