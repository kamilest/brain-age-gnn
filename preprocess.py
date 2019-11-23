"""
Graph preprocessing file.
Collects features and assembles the population graph.


Collects the relevant timeseries, 
computes functional/structural connectivity matrices
computes graph adjacency scores
connects nodes into a graph, assigning collected features
"""

import numpy as np
import pandas as pd
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


def get_ts_filenames(num_subjects=None, randomise=True, seed=0):
    ts_filenames = [f for f in sorted(os.listdir(data_timeseries))]

    if num_subjects is not None:
        if randomise:
            np.random.seed(seed)
            return np.random.choice(ts_filenames, num_subjects)
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

    return [f[:-len("_ts_raw.txt")] for f in get_ts_filenames(num_subjects, randomise, seed)]


# TODO: include the argument for the kind of connectivity matrix (partial correlation, correlation, lasso,...)
def get_functional_connectivity(subject_id):
    """
    Returns the correlation matrix for the parcellated timeseries data, precomputing if necessary.

    Args:
        subject_id: ID of subject.

    Returns:
        The flattened lower triangle of the correlation matrix for the parcellated timeseries data.
    """
    if subject_id not in os.listdir(data_precomputed_fcms):
        precompute.precompute_fcm(subject_id)

    return np.load(os.path.join(data_precomputed_fcms, subject_id + '.npy'))


def get_structural_data(subject_ids):
    """
    Retrieves the non-timeseries data for the list of subjects.

    Args:
        subject_ids: List of subject IDs.

    Returns: The matrix containing the structural attributes for the list of subjects, of shape (num_subjects,
    num_structural_attributes)
    """

    # Retrieve cortical thickness.
    cts = pd.read_csv(data_ct, sep=',')
    subject_cts = cts.where(cts['\"NewID\"'] in subject_ids)

    # Retrieve Euler indices
    eids = pd.read_csv(data_euler, sep=',')
    subject_eids = eids.where(eids['eid'] in subject_ids)

    # Merge dataframes.

    return None


def get_similarity(subject_i, subject_j):
    """
    Computes the similarity score between two subjects.

    Args:
        subject_i: First subject.
        subject_j: Second subject.

    Returns:
        Similarity score.
    """

    return np.random.rand()


def construct_edge_list(subject_ids, similarity_threshold=0.5):
    """
    Constructs the adjacency list of the population graph based on the
    similarity metric.
  
    Args:
        subject_ids: List of subject IDs.
        similarity_threshold: The threshold above which the edge should be added.

    Returns:
        Graph connectivity in coordinate format of shape [2, num_edges]. The
        same edge (v, w) appears twice as (v, w) and (w, v) to represent
        bidirectionality.
    """
    v_list = []
    w_list = []

    for i, id_i in enumerate(subject_ids):
        for j, id_j in enumerate(subject_ids):
            if get_similarity(id_i, id_j) > similarity_threshold:
                v_list.extend([i, j])
                w_list.extend([j, i])

    return [v_list, w_list]


def construct_population_graph(size, save=True, save_dir=graph_root):
    subject_ids = get_subject_ids(size)
    connectivities = [get_functional_connectivity(i) for i in subject_ids]

    edge_index = torch.tensor(
        construct_edge_list(subject_ids),
        dtype=torch.long)

    # Take the first 90% to train, 10% to test
    split = int(size * 0.9)
    train_mask = subject_ids[:split]
    test_mask = subject_ids[-(size-split):]

    population_graph = Data(
        x=connectivities,
        edge_index=edge_index,
        y=None,
        train_mask=train_mask,
        test_mask=test_mask
    )

    if save:
        torch.save(population_graph, os.path.join(save_dir, 'population_graph.pt'))

    return population_graph


def load_population_graph(graph_root):
    return torch.load(os.path.join(graph_root, 'population_graph.pt'))
