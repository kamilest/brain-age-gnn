"""
Graph preprocessing file.
Collects features and assembles the population graph.


Collects the relevant timeseries, 
computes functional/structural connectivity matrices
computes graph adjacency scores
connects nodes into a graph, assigning collected features
"""

import numpy as np
import os

import torch
from torch_geometric.data import Data

import sklearn
from sklearn.preprocessing import OneHotEncoder

import precompute

# Data sources.
data_timeseries = 'data/raw_ts'
data_precomputed_fcms = 'data/processed_ts'
data_phenotype = 'data/phenotype.csv'
graph_root = 'data/graph'

# Graph construction phenotypic parameters.
# http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=31
SEX_UID = '31-0.0'
# http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21003
AGE_UID = '21003-2.0'

# Exclude the following raw timeseries due to incorrect size.
EXCLUDED_UKBS = ['UKB2203847_ts_raw.txt', 'UKB2208238_ts_raw.txt', 'UKB2697888_ts_raw.txt']

def get_ts_filenames(num_subjects=None, randomise=True, seed=0):
    ts_filenames = [f for f in sorted(os.listdir(data_timeseries))]
    for patient in EXCLUDED_UKBS:
        if patient in ts_filenames:
            print('Excluded ', patient)
            ts_filenames.remove(patient)

    if num_subjects is not None:
        if randomise:
            np.random.seed(seed)
            return np.random.choice(ts_filenames, num_subjects, replace=False)
        else:
            return ts_filenames[:num_subjects]
    else:
        return ts_filenames


def get_subject_ids(num_subjects=None, randomise=True, seed=0):
    """
    Gets the list of subject IDs for a spcecified number of subjects. If the number of subjects is not specified, all
    IDs are returned.
  
    Args:
        num_subjects: The number of subjects.
        randomise: Indicates whether to use a random seed for selection of subjects.
        seed: Seed value.

    Returns:
        List of subject IDs.
    """

    return sorted([f[:-len("_ts_raw.txt")] for f in get_ts_filenames(num_subjects, randomise, seed)])


# TODO: include the argument for the kind of connectivity matrix (partial correlation, correlation, lasso,...)
def get_functional_connectivity(subject_id):
    """
    Returns the correlation matrix for the parcellated timeseries data, precomputing if necessary.

    Args:
        subject_id: ID of subject.

    Returns:
        The flattened lower triangle of the correlation matrix for the parcellated timeseries data.
    """
    if subject_id + '.npy' not in os.listdir(data_precomputed_fcms):
        precompute.precompute_fcm(subject_id)

    return np.load(os.path.join(data_precomputed_fcms, subject_id + '.npy'))


def get_all_functional_connectivities(subject_ids):
    connectivities = []
    exclude = []
    for i, subject_id in enumerate(subject_ids):
        connectivity = get_functional_connectivity(subject_id)
        if len(connectivity) != 70500:
            exclude.append(i)
            print('Excluded {}: connectivity matrix length {}'.format(subject_id, len(connectivity)))
        else:
            connectivities.append(connectivity)

    return connectivities, np.delete(subject_ids, exclude)


def functional_connectivities_pca(connectivities, train_idx, random_state=0):
    connectivity_pca = sklearn.decomposition.PCA(random_state=random_state)
    connectivity_pca.fit(connectivities[train_idx])
    return connectivity_pca.transform(connectivities)


def get_similarity(phenotypes, subject_i, subject_j):
    """
    Computes the similarity score between two subjects.

    Args:
        phenotypes: Dataframe with phenotype values.
        subject_i: First subject.
        subject_j: Second subject.

    Returns:
        Similarity score.
    """
    return int(phenotypes.loc[subject_i, SEX_UID] == phenotypes.loc[subject_j, SEX_UID])


def construct_edge_list(phenotypes, similarity_function=get_similarity, similarity_threshold=0.5):
    """
    Constructs the adjacency list of the population graph based on a similarity metric provided.
  
    Args:
        phenotypes: Dataframe with phenotype values.
        similarity_function: Function which is returns similarity between two subjects according to some metric.
        similarity_threshold: The threshold above which the edge should be added.

    Returns:
        Graph connectivity in coordinate format of shape [2, num_edges]. The
        same edge (v, w) appears twice as (v, w) and (w, v) to represent
        bidirectionality.
    """
    v_list = []
    w_list = []

    for i, id_i in enumerate(phenotypes.index):
        iter_j = iter(enumerate(phenotypes.index))
        [next(iter_j) for _ in range(i + 1)]
        for j, id_j in iter_j:
            if similarity_function(phenotypes, id_i, id_j) > similarity_threshold:
                v_list.extend([i, j])
                w_list.extend([j, i])

    return [v_list, w_list]


def construct_population_graph(size=None,
                               save=True,
                               save_dir=graph_root,
                               name=None,
                               functional=False,
                               pca=False,
                               structural=True,
                               euler=True):
    if name is None:
        name = 'population_graph_' \
               + (str(size) if size is not None else 'all') \
               + ('_functional' if functional else '') \
               + ('_PCA' if functional and pca else '') \
               + ('_structural' if structural else '') \
               + ('_euler' if euler else '') \
               + '.pt'

    subject_ids = get_subject_ids(size)

    phenotypes = precompute.extract_phenotypes([SEX_UID, AGE_UID], subject_ids)
    subject_ids = phenotypes.index

    if functional:
        functional_connectivities, subject_ids = get_all_functional_connectivities(subject_ids)
    else:
        functional_connectivities = []

    if structural:
        ct = precompute.extract_cortical_thickness(subject_ids)
        subject_ids = ct.index
    else:
        ct = []

    if euler:
        euler = precompute.extract_euler(subject_ids)
        subject_ids = euler.index
    else:
        euler = []

    print('{} subjects remaining for graph construction.'.format(len(subject_ids)))

    # sex = OneHotEncoder().fit_transform(phenotypes[SEX_UID].to_numpy().reshape(-1, 1))
    # ct_sex = np.concatenate((ct.to_numpy(), sex.toarray()), axis=1)
    # if euler:
    #
    # else:
    # connectivities = ct_sex

    labels = torch.tensor([phenotypes[AGE_UID].tolist()], dtype=torch.float32).transpose_(0, 1)

    edge_index = torch.tensor(
        construct_edge_list(subject_ids),
        dtype=torch.long)

    np.random.seed(0)

    num_subjects = len(subject_ids)

    num_train = int(num_subjects * 0.85)
    num_validate = int(num_subjects * 0.05)

    train_val_idx = np.random.choice(range(num_subjects), num_train + num_validate, replace=False)
    train_idx = np.random.choice(train_val_idx, num_train, replace=False)
    validate_idx = list(set(train_val_idx) - set(train_idx))
    test_idx = list(set(range(num_subjects)) - set(train_val_idx))

    assert (len(np.intersect1d(train_idx, validate_idx)) == 0)
    assert (len(np.intersect1d(train_idx, test_idx)) == 0)
    assert (len(np.intersect1d(validate_idx, test_idx)) == 0)

    train_np = np.zeros(num_subjects, dtype=bool)
    train_np[train_idx] = True

    validate_np = np.zeros(num_subjects, dtype=bool)
    validate_np[validate_idx] = True

    test_np = np.zeros(num_subjects, dtype=bool)
    test_np[test_idx] = True

    train_mask = torch.tensor(train_np, dtype=torch.bool)
    validate_mask = torch.tensor(validate_np, dtype=torch.bool)
    test_mask = torch.tensor(test_np, dtype=torch.bool)

    if functional and pca:
        functional_connectivities = functional_connectivities_pca(functional_connectivities, train_idx)

    fc_tensor = torch.tensor(functional_connectivities, dtype=torch.float32)

    # TODO: scale structural data.
    # TODO: structural/Euler data tensor construction.
    # TODO: concatenate functional, structural, Euler data as needed.
    # if structural:
    #     scaler = sklearn.preprocessing.StandardScaler()
    #     scaler.fit(connectivities[train_idx])
    #     connectivities_transformed = torch.tensor(scaler.transform(connectivities),
    #                                               dtype=torch.float32)

    population_graph = Data(
        x=functional_connectivities,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        test_mask=test_mask
    )

    population_graph.validate_mask = validate_mask

    if save:
        torch.save(population_graph, os.path.join(save_dir, name))

    return population_graph


def load_population_graph(graph_root, name):
    return torch.load(os.path.join(graph_root, name))


if __name__ == '__main__':
    graph = construct_population_graph(1000)