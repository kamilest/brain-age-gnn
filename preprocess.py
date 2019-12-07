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

import precompute

# Data sources.
data_timeseries = 'data/raw_ts'
data_precomputed_fcms = 'data/processed_ts'
data_phenotype = 'data/phenotype.csv'
data_ct = 'data/CT.csv'
data_euler = 'data/Euler.csv'
graph_root = 'data/graph'

# Graph construction phenotypic parameters.
# http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=31
SEX_UID = '31-0.0'
# http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21003
AGE_UID = '21003-2.0'


def get_ts_filenames(num_subjects=None, randomise=True, seed=0):
    ts_filenames = [f for f in sorted(os.listdir(data_timeseries))]

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


def extract_connectivities(subject_ids):
    connectivities = []
    exclude = []
    for i, subject_id in enumerate(subject_ids):
        connectivity = get_functional_connectivity(subject_id)
        if len(connectivity) != 70500:
            exclude.append(i)
        else:
            connectivities.append(connectivity)

    print('Additional {} entries removed due to small connectivity matrices.'.format(len(exclude)))
    return connectivities, np.delete(subject_ids, exclude), subject_ids[exclude]


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


def construct_edge_list(phenotypes, similarity_threshold=0.5):
    """
    Constructs the adjacency list of the population graph based on the
    similarity metric.
  
    Args:
        phenotypes: Dataframe with phenotype values.
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
        [next(iter_j) for _ in range(i+1)]
        for j, id_j in iter_j:
            if get_similarity(phenotypes, id_i, id_j) > similarity_threshold:
                v_list.extend([i, j])
                w_list.extend([j, i])

    return [v_list, w_list]


def construct_population_graph(size=None, save=True, save_dir=graph_root, name='population_graph.pt'):
    subject_ids = get_subject_ids(size)
    print(subject_ids)

    phenotypes = precompute.extract_phenotypes([SEX_UID, AGE_UID], subject_ids)
    connectivities = torch.tensor([get_functional_connectivity(i) for i in phenotypes.index], dtype=torch.float32)

    labels = torch.tensor([phenotypes[AGE_UID].tolist()], dtype=torch.float32).transpose_(0, 1)

    edge_index = torch.tensor(
        construct_edge_list(phenotypes),
        dtype=torch.long)

    np.random.seed(0)
    num_train = int(len(phenotypes) * 0.9)
    split_mask = np.zeros(len(phenotypes), dtype=bool)
    split_mask[np.random.choice(len(phenotypes), num_train, replace=False)] = True

    train_mask = torch.tensor(split_mask, dtype=torch.bool)
    test_mask = torch.tensor(np.invert(split_mask), dtype=torch.bool)

    population_graph = Data(
        x=connectivities,
        edge_index=edge_index,
        y=labels,
        train_mask=train_mask,
        test_mask=test_mask
    )

    if save:
        torch.save(population_graph, os.path.join(save_dir, name))

    return population_graph


def load_population_graph(graph_root, name):
    return torch.load(os.path.join(graph_root, name))


if __name__ == '__main__':
    construct_population_graph(name='population_graph1000.pt')
    graph = load_population_graph(graph_root, name='population_graph1000.pt')
