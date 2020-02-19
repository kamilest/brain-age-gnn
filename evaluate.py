import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from preprocess import concatenate_graph_features


def test_subject_split(train_idx, validate_idx, test_idx):
    assert (len(np.intersect1d(train_idx, validate_idx)) == 0)
    assert (len(np.intersect1d(train_idx, test_idx)) == 0)
    assert (len(np.intersect1d(validate_idx, test_idx)) == 0)


def get_random_subject_split(population_graph, test=0.1, seed=0):
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
    features = concatenate_graph_features(population_graph)
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
    features = concatenate_graph_features(population_graph)
    labels = population_graph.y.numpy()

    train_test_split = StratifiedKFold(n_splits=n_folds, random_state=random_state)
    folds = []
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

            folds.append([train_idx, validate_idx, test_idx])

    return folds


def set_training_masks(population_graph, train_index, validate_index, test_index):
    train_mask, validate_mask, test_mask = get_subject_split_masks(train_index, validate_index, test_index)
    population_graph.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    population_graph.validate_mask = torch.tensor(validate_mask, dtype=torch.bool)
    population_graph.test_mask = torch.tensor(test_mask, dtype=torch.bool)


def get_subject_split_masks(train_index, validate_index, test_index):
    num_subjects = len(train_index) + len(validate_index) + len(test_index)

    train_mask = np.zeros(num_subjects, dtype=bool)
    train_mask[train_index] = True

    validate_mask = np.zeros(num_subjects, dtype=bool)
    validate_mask[validate_index] = True

    test_mask = np.zeros(num_subjects, dtype=bool)
    test_mask[test_index] = True

    return train_mask, validate_mask, test_mask


def add_population_graph_noise(population_graph, p, noise_amplitude=0.05):
    """Adds white Gaussian noise to the nodes of the population graph.

    :param population_graph: population graph.
    :param p: proportion of training nodes with added noise.
    :param noise_amplitude: the variance of white noise.
    """

    nodes = population_graph.x.numpy().copy()
    train_idx = np.where(population_graph.train_mask.numpy())[0]
    noisy_train_idx = np.random.choice(train_idx, round(len(train_idx) * p), replace=False)

    for i in noisy_train_idx:
        nodes[i] += np.random.normal(0, noise_amplitude, len(nodes[i]))

    population_graph.x = torch.tensor(nodes)


def remove_population_graph_edges(population_graph, p):
    """Removes graph edges with probability p.

    :param population_graph: path to the population graph file.
    :param p: proportion of the edges removed.
    """
    edges = np.transpose(population_graph.edge_index.numpy().copy())
    unique_edges = np.unique([list(x) for x in (frozenset(y) for y in edges)], axis=0)

    idx = np.random.choice(range(len(unique_edges)), replace=False, size=round(len(unique_edges) * p))
    unique_edges = np.delete(unique_edges, idx, axis=0)

    v_list, w_list = [], []
    for edge in unique_edges:
        v_list.extend([edge[0], edge[1]])
        w_list.extend([edge[1], edge[0]])

    # for i in range(round(len(unique_edges) * p)):
    #     idx = np.random.randint(0, len(edges))
    #     unique_edges = np.delete(unique_edges, idx, axis=0)

    population_graph.edge_index = torch.tensor([v_list, w_list], dtype=torch.long)


def decrease_population_graph_train_set(graph, test_set_sizes):
    """Decreases the training set (more unlabeled nodes).

    :param graph: path to the population graph file.
    :param test_set_sizes:
    :return:
    """


def measure_predictive_power_drop():
    """Measures the drop in performance metrics with increased noise or more missing data.

    :return: the range of values at different modification levels.
    """
    pass

