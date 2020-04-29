"""
Graph neural network training component.

Splits subjects in the population graph into training, validation, and test sets, stratifying by sex and age.
Sets training masks in the population graph.
Provides methods for training the graph neural network (potentially with cross-validation).
"""


import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import graph_transform
from brain_gnn import BrainGCN, BrainGAT, ConvTypes
from ukb_preprocess import SIMILARITY_LOOKUP, ICD10_LOOKUP

graph_root = 'data/graph'
model_root = 'data/model'

GRAPH_NAMES = sorted(os.listdir(graph_root))

# Used by wandb framework.
hyperparameter_defaults = dict(
    model='gcn',
    epochs=5000,
    learning_rate=5e-4,
    dropout=0,
    weight_decay=0,
    n_conv_layers=1,
    layer_sizes=[364, 364, 512, 256, 1]
)


def get_confounding_features(population_graph):
    """
    Returns the dataframe of confounding non-imaging features used for further stratification.

    :param population_graph: population graph.
    :return: the dataframe with confounding features against which to stratify (in this case age and sex).
    """
    similarity_lookup = pd.read_pickle(SIMILARITY_LOOKUP).loc[
        population_graph.subject_index, ['AGE', 'SEX']].fillna(-1)
    icd10_lookup = pd.read_pickle(ICD10_LOOKUP).loc[population_graph.subject_index].fillna(-1)

    labels = np.hstack(
        [a.reshape(population_graph.num_nodes, -1) for a in [similarity_lookup.to_numpy(), icd10_lookup.to_numpy()]])

    return similarity_lookup.to_numpy()


def get_encoded_confounding_features(population_graph):
    """
    Encodes the confounding features as labels.

    :param population_graph: population graph.
    :return: new labels of population graph subjects containing all the confounding variables.
    """

    confounding_features = get_confounding_features(population_graph)
    return LabelEncoder().fit_transform(["".join(a) for a in confounding_features.astype(str)])


def test_subject_split(train_idx, validate_idx, test_idx):
    """Tests subject split, asserting whether no subjects spill between the splits.

    :param train_idx: subjects in train set
    :param validate_idx: subjects in validation set
    :param test_idx: subjects in test set
    """

    assert (len(np.intersect1d(train_idx, validate_idx)) == 0)
    assert (len(np.intersect1d(train_idx, test_idx)) == 0)
    assert (len(np.intersect1d(validate_idx, test_idx)) == 0)


def get_random_subject_split(population_graph, test=0.1, seed=0):
    """Returns a random subject split for a population graph.

    :param population_graph: population graph object.
    :param test: proportion of subjects to remain in test set.
    :param seed: random seed for splitting.
    :return a tuple of three lists, of integer array indexes for train, validation and test sets. The order follows the
        subject order in the population graph.
    """

    num_subjects = population_graph.num_nodes
    np.random.seed(seed)

    assert 0 <= test <= 1
    train_validate = 1 - test
    train = 0.9 * train_validate
    validate = 0.1 * train_validate

    num_train = int(num_subjects * train)
    num_validate = int(num_subjects * validate)

    train_val_idx = np.random.choice(range(num_subjects), num_train + num_validate, replace=False)
    train_idx = np.sort(np.random.choice(train_val_idx, num_train, replace=False))
    validate_idx = np.sort(list(set(train_val_idx) - set(train_idx)))
    test_idx = np.sort(list(set(range(num_subjects)) - set(train_val_idx)))

    # test_subject_split(train_idx, validate_idx, test_idx)
    return train_idx.astype(int), validate_idx.astype(int), test_idx.astype(int)


def get_stratified_subject_split(population_graph, test_size=0.1, random_state=0):
    """Returns a subject split into train validation and test sets for a population graph, stratified by label.

    :param population_graph: population graph object.
    :param test_size: proportion of subjects to remain in test set.
    :param random_state: random seed for splitting.
    :return a tuple of three lists, of integer array indexes for train, validation and test sets. The order follows the
        subject order in the population graph.
    """

    features = graph_transform.concatenate_graph_features(population_graph)
    labels = population_graph.y.numpy()

    train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    for train_validate_index, test_index in train_test_split.split(features, labels):
        train_validate_index = np.sort(train_validate_index)
        test_index = np.sort(test_index)
        features_train = features[train_validate_index]
        labels_train = labels[train_validate_index]

        train_validate_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
        for train_index, validate_index in train_validate_split.split(features_train, labels_train):
            train_index = np.sort(train_index)
            validate_index = np.sort(validate_index)

            train_idx = train_validate_index[train_index]
            validate_idx = train_validate_index[validate_index]
            test_idx = test_index

            test_subject_split(train_idx, validate_idx, test_idx)
            return train_idx, validate_idx, test_idx


def get_cv_subject_split(population_graph, n_folds=10, random_state=0):
    """Get the cross-validated subject split for the a given population graph and the number of folds.

    :param population_graph: population graph object.
    :param n_folds: number of folds in which to split the subjects in the population graph.
    :param random_state: random seed for splitting.
    :return a list of folds, each consisting of lists of array indices for training, validation and test sets.
    """

    labels = get_encoded_confounding_features(population_graph)
    features = np.zeros(population_graph.num_nodes)

    train_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=random_state)
    folds = []
    for train_validate_index, test_index in train_test_split.split(features, labels):
        train_validate_index = np.sort(train_validate_index)
        test_index = np.sort(test_index)
        features_train = features[train_validate_index]
        labels_train = labels[train_validate_index]

        train_validate_split = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for train_index, validate_index in train_validate_split.split(features_train, labels_train):
            train_index = np.sort(train_index)
            validate_index = np.sort(validate_index)

            train_idx = train_validate_index[train_index]
            validate_idx = train_validate_index[validate_index]
            test_idx = test_index
            test_subject_split(train_idx, validate_idx, test_idx)

            folds.append([train_idx, validate_idx, test_idx])

    return folds


def set_training_masks(population_graph, train_index, validate_index, test_index, ignore_nonhealthy=True):
    """Modifies the population graph by setting the boolean masks for the given train, validation and test set indices.
    Optionally masks the subjects that are considered to have unhealthy brains.

    :param population_graph: population graph object.
    :param train_index: indices of subjects belonging to the train set.
    :param validate_index: indices of subjects belonging to the validation set.
    :param test_index: indices of subjects belonging to the test set.
    :param ignore_nonhealthy: if set to True, additionally masks subjects with mental/nervous system problems.
    """

    train_mask, validate_mask, test_mask = get_subject_split_masks(train_index, validate_index, test_index)

    if ignore_nonhealthy:
        population_graph.full_train_mask = torch.tensor(train_mask, dtype=torch.bool)
        population_graph.train_mask = torch.tensor(
            np.logical_and(train_mask, population_graph.brain_health_mask), dtype=torch.bool)
        population_graph.full_validate_mask = torch.tensor(validate_mask, dtype=torch.bool)
        population_graph.validate_mask = torch.tensor(
            np.logical_and(validate_mask, population_graph.brain_health_mask), dtype=torch.bool)
        population_graph.full_test_mask = torch.tensor(test_mask, dtype=torch.bool)
        population_graph.test_mask = torch.tensor(
            np.logical_and(test_mask, population_graph.brain_health_mask), dtype=torch.bool)

    else:
        population_graph.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        population_graph.validate_mask = torch.tensor(validate_mask, dtype=torch.bool)
        population_graph.test_mask = torch.tensor(test_mask, dtype=torch.bool)


def get_subject_split_masks(train_index, validate_index, test_index):
    """Returns the boolean masks for the arrays of integer indices.

    :param train_index: indices of subjects in the train set.
    :param validate_index: indices of subjects in the validation set.
    :param test_index: indices of subjects in the test set.
    :return a tuple of boolean masks corresponding to the train/validate/test set indices.
    """

    num_subjects = len(train_index) + len(validate_index) + len(test_index)

    train_mask = np.zeros(num_subjects, dtype=bool)
    train_mask[train_index] = True

    validate_mask = np.zeros(num_subjects, dtype=bool)
    validate_mask[validate_index] = True

    test_mask = np.zeros(num_subjects, dtype=bool)
    test_mask[test_index] = True

    return train_mask, validate_mask, test_mask

def train(conv_type, graph, device, n_conv_layers=0, layer_sizes=None, epochs=3500, lr=0.005, dropout_p=0,
          weight_decay=1e-5, log=True, early_stopping=True, patience=10, delta=0.005, cv=False, fold=0,
          run_name=None, min_epochs=1000):
    """
    Trains the graph neural network on a population graph.

    :param conv_type: convolution type, ConvType.GCN or ConvType.GAT
    :param graph: the population graph
    :param device: device on which the neural network will be trained. 'cpu' or 'cuda:x'
    :param n_conv_layers: number of convolutional layers in GNN
    :param layer_sizes: array of layer sizes, first n_conv_layers of which convolutional, the rest fully connected
    :param epochs: number of epochs for which to train
    :param lr: learning rate
    :param dropout_p: dropout probability (probability of killing the unit)
    :param weight_decay: weight decay
    :param log: indicates whether to log results to wandb.
    :param early_stopping: indicates whether to use early stopping.
    :param patience: indicates how long to tolerate no improvement in MSE before early stopping.
    :param delta: defines the threshold of how much the model has to improve in `patience` epochs.
    :param cv: whether this is part of cross-validation training (used for logging).
    :param fold: in case this is cross-validation training, which fold is this (used for logging).
    :param run_name: wandb run name (used for smoother logging when cross-validation is used).
    :param min_epochs: minimum number of epochs that have to pass before early stopping is enabled.
    :return: the model and (run name, validation set predictions vs actual labels).
    """

    data = graph.to(device)
    assert n_conv_layers >= 0

    if layer_sizes is None:
        layer_sizes = []

    if ConvTypes(conv_type) == ConvTypes.GCN:
        model = BrainGCN(graph.num_node_features, n_conv_layers, layer_sizes, dropout_p).to(device)
    else:
        model = BrainGAT(graph.num_node_features, n_conv_layers, layer_sizes, dropout_p).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Saving (early sopping) checkpoints
    early_stopping_checkpoint = 'checkpoint.pt'

    # Initialise early stopping.
    early_stopping_count = 0
    early_stopping_min_val_loss = None
    early_stop = False

    # Initialise wandb log.
    if log:
        wandb.run.save()
        wandb.watch(model)
        wandb.config.update({
            "graph_name": graph.name,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay})

        if not cv or run_name is None:
            run_name = wandb.run.name
        else:
            wandb.run.name = run_name + '-fold-{}'.format(fold)
        wandb.run.save()
        early_stopping_checkpoint = '{}_{}_state_dict.pt'.format(
            datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            wandb.run.name)

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        train_mse = loss.item(),
        train_r2 = r2_score(data.y[data.train_mask].cpu().detach().numpy(),
                            out[data.train_mask].cpu().detach().numpy())
        train_r = pearsonr(data.y[data.train_mask].cpu().detach().numpy().flatten(),
                           out[data.train_mask].cpu().detach().numpy().flatten())[0]
        val_mse = F.mse_loss(out[data.validate_mask], data.y[data.validate_mask]).item()
        val_r2 = r2_score(data.y[data.validate_mask].cpu().detach().numpy(),
                          out[data.validate_mask].cpu().detach().numpy()),
        val_r = pearsonr(data.y[data.validate_mask].cpu().detach().numpy().flatten(),
                         out[data.validate_mask].cpu().detach().numpy().flatten())[0]

        if early_stopping:
            if early_stopping_min_val_loss is None:
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, early_stopping_checkpoint))
                early_stopping_min_val_loss = val_mse
            elif val_mse > early_stopping_min_val_loss - delta and epoch > min_epochs:
                early_stopping_count += 1
                if early_stopping_count >= patience:
                    early_stop = True
            else:
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, early_stopping_checkpoint))
                if log:
                    wandb.save(early_stopping_checkpoint)
                early_stopping_min_val_loss = val_mse
                early_stopping_count = 0

        if log:
            wandb.log({"train_mse_fold_{}".format(fold): train_mse[0],
                       "train_r2_fold_{}".format(fold): train_r2,
                       "train_r_fold_{}".format(fold): train_r,
                       "validation_mse_fold_{}".format(fold): val_mse,
                       "validation_r2_fold_{}".format(fold): val_r2[0],
                       "validation_r_fold_{}".format(fold): val_r})
        print(epoch, train_mse, train_r2, train_r)
        print(epoch, val_mse, val_r2, val_r)
        print()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        if early_stop:
            break

    # Load the best early stopping model.
    if early_stopping:
        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, early_stopping_checkpoint)))

    model.eval()
    final_model = model(data)
    predicted = final_model[data.validate_mask].cpu()
    actual = data.y[data.validate_mask].cpu()
    final_r2 = r2_score(actual.detach().numpy(), predicted.detach().numpy())
    final_r = pearsonr(actual.detach().numpy().flatten(), predicted.detach().numpy().flatten())
    print('Final validation r2: {}'.format(final_r2))
    print('Final validation MSE: {}'.format(F.mse_loss(predicted, actual)))
    print('Final Pearson\'s r: {}'.format(final_r))

    if log:
        wandb.run.summary["final_validation_mse_fold_{}".format(fold)] = F.mse_loss(predicted, actual)
        wandb.run.summary["final_validation_r2_fold_{}".format(fold)] = final_r2
        wandb.run.summary["final_validation_r_fold_{}".format(fold)] = final_r

    return model, (run_name, predicted, actual)


def train_with_cross_validation(conv_type, graph, device, n_folds=10, n_conv_layers=0, layer_sizes=None, epochs=350,
                                lr=0.005, dropout_p=0, weight_decay=1e-5, log=True, early_stopping=True,
                                patience=10, delta=0.005):
    """
    Trains the graph neural network on a population graph.

    :param conv_type: convolution type, ConvType.GCN or ConvType.GAT
    :param graph: the population graph
    :param device: device on which the neural network will be trained. 'cpu' or 'cuda:x'
    :param n_conv_layers: number of convolutional layers in GNN
    :param layer_sizes: array of layer sizes, first n_conv_layers of which convolutional, the rest fully connected
    :param epochs: number of epochs for which to train
    :param lr: learning rate
    :param dropout_p: dropout probability (probability of killing the unit)
    :param weight_decay: weight decay
    :param log: indicates whether to log results to wandb.
    :param early_stopping: indicates whether to use early stopping.
    :param patience: indicates how long to tolerate no improvement in MSE before early stopping.
    :param delta: defines the threshold of how much the model has to improve in `patience` epochs.
    :return: array of results as in train method, repeated for all folds.
    """

    folds = get_cv_subject_split(graph, n_folds=n_folds)
    results = []
    run_name = None

    wandb.init(project="brain-age-gnn", config=hyperparameter_defaults, reinit=True)
    wandb.save("*.pt")

    for i, fold in enumerate(folds):
        set_training_masks(graph, *fold)
        graph_transform.graph_feature_transform(graph)

        _, fold_result = train(conv_type, graph, device, n_conv_layers=n_conv_layers, layer_sizes=layer_sizes,
                               epochs=epochs, lr=lr, dropout_p=dropout_p, weight_decay=weight_decay, log=log,
                               early_stopping=early_stopping, patience=patience, delta=delta, cv=True, fold=i,
                               run_name=run_name)
        run_name = fold_result[0]
        fold_scores = fold_result[1:]
        results.append(fold_scores)
        if np.mean([F.mse_loss(predicted, actual).item() for predicted, actual in results], axis=0) > 30:
            break

    # Add cross-validation summaries to the last fold
    cv_mse = [F.mse_loss(predicted, actual).item() for predicted, actual in results]
    cv_r2 = [r2_score(actual.detach().numpy(), predicted.detach().numpy()) for predicted, actual in results]
    cv_r = [pearsonr(actual.detach().numpy().flatten(), predicted.detach().numpy().flatten())
            for predicted, actual in results]
    wandb.run.summary["cv_validation_average_mse"] = np.mean(cv_mse, axis=0)
    wandb.run.summary["cv_validation_average_r2"] = np.mean(cv_r2, axis=0)
    wandb.run.summary["cv_validation_average_r"] = np.mean(cv_r, axis=0)

    return results
