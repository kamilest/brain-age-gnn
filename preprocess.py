"""
Graph preprocessing file.
Collects features and assembles the population graph.


Collects the relevant timeseries, 
computes functional/structural connectivity matrices
computes graph adjacency scores
connects nodes into a graph, assigning collected features
"""

import csv
import os
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from torch_geometric.data import Data

import precompute
import similarity
from phenotype import Phenotype

# Data sources.
data_root = 'data'
data_timeseries = 'data/raw_ts'
data_precomputed_fcms = 'data/processed_ts'
data_phenotype = 'data/phenotype.csv'
graph_root = 'data/graph'

# Graph construction phenotypic parameters.
AGE_UID = Phenotype.get_biobank_codes(Phenotype.AGE)[0]


def get_subject_ids(num_subjects=None, randomise=True, seed=0):
    """Gets the list of subject IDs for a spcecified number of subjects.

    :param num_subjects: number of subjects.
    :param randomise: indicates whether to use a random seed for selection of subjects.
    :param seed: seed value.
    :return: list of subject IDs.
    """

    if not os.path.isfile(os.path.join(data_root, 'subject_ids.npy')):
        precompute.precompute_subject_ids()

    subject_ids = np.load(os.path.join(data_root, 'subject_ids.npy'), allow_pickle=True)

    if not num_subjects:
        return subject_ids

    if randomise:
        np.random.seed(seed)
        return np.random.choice(subject_ids, num_subjects, replace=False)
    else:
        return subject_ids[:num_subjects]


# TODO: include the argument for the kind of connectivity matrix (partial correlation, correlation, lasso,...)
def get_functional_connectivity(subject_id):
    """Returns the correlation matrix for the parcellated timeseries data.
    If necessary precomputes the matrix.

    :param subject_id: subject ID.
    :return: the flattened lower triangle of the correlation matrix for the parcellated timeseries data.
    """

    if subject_id + '.npy' not in os.listdir(data_precomputed_fcms):
        precompute.precompute_fcm(subject_id)

    return np.load(os.path.join(data_precomputed_fcms, subject_id + '.npy'))


def get_all_functional_connectivities(subject_ids):
    connectivities = []
    for i, subject_id in enumerate(subject_ids):
        connectivity = get_functional_connectivity(subject_id)
        assert len(connectivity) == 70500
        connectivities.append(connectivity)

    return connectivities


def functional_connectivities_pca(connectivities, train_idx, random_state=0):
    connectivity_pca = sklearn.decomposition.PCA(random_state=random_state)
    connectivity_pca.fit(connectivities[train_idx])
    return connectivity_pca.transform(connectivities)


def test_subject_split(train_idx, validate_idx, test_idx):
    assert (len(np.intersect1d(train_idx, validate_idx)) == 0)
    assert (len(np.intersect1d(train_idx, test_idx)) == 0)
    assert (len(np.intersect1d(validate_idx, test_idx)) == 0)


def get_random_subject_split(num_subjects, test=0.1, seed=0):
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

    test_subject_split(train_idx, validate_idx, test_idx)
    return train_idx, validate_idx, test_idx


def get_stratified_subject_split(features, labels, test_size=0.1, random_state=0):
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


def get_cv_subject_split(features, labels, n_folds=10, random_state=0):
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


def get_subject_split(features, labels, stratify):
    if stratify:
        stratified_subject_split = get_stratified_subject_split(features, labels)
        train_mask, validate_mask, test_mask = get_subject_split_masks(*stratified_subject_split)
    else:
        subject_split = get_random_subject_split(len(features))
        train_mask, validate_mask, test_mask = get_subject_split_masks(*subject_split)

    return train_mask, validate_mask, test_mask


def get_subject_split_masks(train_index, validate_index, test_index):
    num_subjects = len(train_index) + len(validate_index) + len(test_index)

    train_mask = np.zeros(num_subjects, dtype=bool)
    train_mask[train_index] = True

    validate_mask = np.zeros(num_subjects, dtype=bool)
    validate_mask[validate_index] = True

    test_mask = np.zeros(num_subjects, dtype=bool)
    test_mask[test_index] = True

    return train_mask, validate_mask, test_mask


def get_graph_name(size, functional, pca, structural, euler, similarity_feature_set):
    separator = '_'
    similarity_feature_string = separator.join([feature.value for feature in similarity_feature_set])
    return 'population_graph_' \
           + (str(size) + '_' if size is not None else 'all_') \
           + similarity_feature_string \
           + ('_functional' if functional else '') \
           + ('_PCA' if functional and pca else '') \
           + ('_structural' if structural else '') \
           + ('_euler' if euler else '') \
           + '.pt'


def collect_graph_data(subject_ids, functional, structural, euler):
    phenotypes = precompute.extract_phenotypes(subject_ids)
    assert len(np.intersect1d(subject_ids, phenotypes.index)) == len(subject_ids)

    if functional:
        functional_data = get_all_functional_connectivities(subject_ids)
    else:
        functional_data = pd.DataFrame(pd.np.empty((len(subject_ids), 0)))

    if structural:
        cortical_thickness_data = precompute.extract_structural(subject_ids, type='cortical_thickness')
        assert len(np.intersect1d(subject_ids, cortical_thickness_data.index)) == len(subject_ids)

        surface_area_data = precompute.extract_structural(subject_ids, type='surface_area')
        assert len(np.intersect1d(subject_ids, surface_area_data.index)) == len(subject_ids)

        volume_data = precompute.extract_structural(subject_ids, type='volume')
        assert len(np.intersect1d(subject_ids, volume_data.index)) == len(subject_ids)
    else:
        cortical_thickness_data = pd.DataFrame(pd.np.empty((len(subject_ids), 0)))
        surface_area_data = pd.DataFrame(pd.np.empty((len(subject_ids), 0)))
        volume_data = pd.DataFrame(pd.np.empty((len(subject_ids), 0)))

    if euler:
        euler_data = precompute.extract_euler(subject_ids)
        assert len(np.intersect1d(subject_ids, euler_data.index)) == len(subject_ids)
    else:
        euler_data = pd.DataFrame(pd.np.empty((len(subject_ids), 0)))

    return {'phenotypes': phenotypes,
            'functional': functional_data,
            'cortical_thickness': cortical_thickness_data,
            'surface_area': surface_area_data,
            'volume': volume_data,
            'euler': euler_data}


def get_sufficient_age_occurrence_index(phenotypes):
    age_counts = phenotypes[AGE_UID].value_counts()
    ages = age_counts.iloc[np.argwhere(age_counts >= 3).flatten()].index.tolist()
    age_index = np.where(phenotypes[AGE_UID].isin(ages))[0]
    return age_index


def construct_edge_list(subject_ids, similarity_function, similarity_threshold=0.5, save=False, graph_name=None):
    """Constructs the adjacency list of the population population_graph based on a similarity metric provided.

    :param subject_ids: subject IDs.
    :param similarity_function: function which is returns similarity between two subjects according to some metric.
    :param similarity_threshold: the threshold above which the edge should be added.
    :param save: inidicates whether to save the population_graph in the logs directory.
    :param graph_name: population_graph name for saved file if population_graph edges are logged.
    :return: population_graph connectivity in coordinate format of shape [2, num_edges].
    The same edge (v, w) appears twice as (v, w) and (w, v) to represent bidirectionality.
    """

    v_list = []
    w_list = []

    if save:
        if graph_name is None:
            graph_name = 'population_graph.csv'

        with open(os.path.join('logs', graph_name), 'w+', newline='') as csvfile:
            wr = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for i, id_i in enumerate(subject_ids):
                wr.writerow([i, i])  # ensure singletons appear in the population_graph adjacency list.
                iter_j = iter(enumerate(subject_ids))
                [next(iter_j) for _ in range(i + 1)]
                for j, id_j in iter_j:
                    if similarity_function(id_i, id_j) > similarity_threshold:
                        wr.writerow([i, j])
                        v_list.extend([i, j])
                        w_list.extend([j, i])
    else:
        for i, id_i in enumerate(subject_ids):
            iter_j = iter(enumerate(subject_ids))
            [next(iter_j) for _ in range(i + 1)]
            for j, id_j in iter_j:
                if similarity_function(id_i, id_j) > similarity_threshold:
                    v_list.extend([i, j])
                    w_list.extend([j, i])

    return [v_list, w_list]


def graph_feature_transform(population_graph, train_mask):
    functional_data, structural_data, euler_data = None, None, None

    # Optional functional data preprocessing (PCA) based on the traning index.
    if population_graph.functional_data is not None:
        functional_data = functional_connectivities_pca(population_graph.functional_data, train_mask)

    # Scaling structural data based on training index.
    # Transforming multiple structural data modalities.
    for structural_feature in population_graph.structural_data.keys():
        if population_graph.structural_data[structural_feature] is not None:
            structural_scaler = sklearn.preprocessing.StandardScaler()
            structural_scaler.fit(population_graph.structural_data[structural_feature][train_mask])
            structural_data = structural_scaler.transform(population_graph.structural_data[structural_feature])

    # Scaling Euler index data based on training index.
    if population_graph.euler_data is not None:
        euler_scaler = sklearn.preprocessing.StandardScaler()
        euler_scaler.fit(population_graph.euler_data[train_mask])
        euler_data = euler_scaler.transform(population_graph.euler_data)

    # Unify feature sets into one feature vector.
    features = np.concatenate([functional_data,
                               structural_data,
                               euler_data], axis=1)

    population_graph.x = torch.tensor(features, dtype=torch.float32)


def construct_population_graph(similarity_feature_set, similarity_threshold=0.5, size=None, functional=False,
                               pca=False, structural=True, euler=True, save=True, logs=True,
                               save_dir=graph_root, name=None):
    if name is None:
        name = get_graph_name(size, functional, pca, structural, euler, similarity_feature_set)

    subject_ids = sorted(get_subject_ids(size))

    # Collect the required data.
    graph_data = collect_graph_data(subject_ids, functional, structural, euler)

    # Remove subjects with too few instances of the label for stratification.
    age_index = get_sufficient_age_occurrence_index(graph_data['phenotypes'])
    subject_ids = sorted(graph_data['phenotypes'].iloc[age_index].index.tolist())
    for feature in graph_data.keys():
        if graph_data[feature] is not None:
            graph_data[feature] = graph_data[feature].iloc[age_index]

    num_subjects = len(subject_ids)
    print('{} subjects remaining for population_graph construction.'.format(num_subjects))

    labels = graph_data['phenotypes'][AGE_UID].to_numpy()
    label_tensor = torch.tensor([labels], dtype=torch.float32).transpose_(0, 1)

    # Construct the edge index.
    similarity_function = similarity.custom_similarity_function(similarity_feature_set)
    edge_index_tensor = torch.tensor(
        construct_edge_list(subject_ids=subject_ids,
                            similarity_function=similarity_function,
                            similarity_threshold=similarity_threshold,
                            save=logs,
                            graph_name=name.replace('.pt', datetime.now().strftime("_%H_%M_%S") + '.csv')),
        dtype=torch.long)

    population_graph = Data()

    population_graph.edge_index = edge_index_tensor
    population_graph.y = label_tensor

    population_graph.subject_index = subject_ids

    population_graph.functional_data = graph_data['functional']
    population_graph.structural_data = {'cortical_thickness': graph_data['cortical_thickness'],
                                        'surface_area': graph_data['surface_area'],
                                        'volume': graph_data['volume']}
    population_graph.euler_data = graph_data['euler']

    if save:
        torch.save(population_graph, os.path.join(save_dir, name))

    return population_graph


def load_population_graph(graph_root, name):
    return torch.load(os.path.join(graph_root, name))


if __name__ == '__main__':
    feature_set = [Phenotype.SEX, Phenotype.FULL_TIME_EDUCATION, Phenotype.FLUID_INTELLIGENCE,
                   Phenotype.PROSPECTIVE_MEMORY_RESULT]
    # TODO restrict similarity threshold.
    graph = construct_population_graph(feature_set, similarity_threshold=0.9, stratify=True, logs=True)

