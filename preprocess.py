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
import scipy.io as sio
import os

from nilearn import connectome

import torch
from torch_geometric.data import Data

# Data sources.
data_root = \
    '/Users/kamilestankeviciute/Google Drive/Part II/Dissertation/' \
    'brain-age-gnn/data'
data_timeseries = os.path.join(data_root, 'data/raw_ts')
data_ct = os.path.join(data_root, 'CT.csv')
data_euler = os.path.join(data_root, 'Euler.csv')
graph_root = os.path.join(data_root, 'graph')


def get_ts_filenames(num_subjects=None):
    ts_filenames = [f for f in sorted(os.listdir(data_timeseries))]

    if num_subjects is not None:
        ts_filenames = ts_filenames[:num_subjects]
    
    return ts_filenames


# TODO: make selection random.
# TODO: consider scalability of this approach when brains don't fit into memory anymore.
def get_subject_ids(num_subjects=None):
    """
    Gets the list of subject IDs for a spcecified number of subjects.
    If the number of subjects is not specified, all IDs are returned.
  
    Args:
        num_subjects: The number of subjects.

    Returns:
        List of subject IDs.
    """

    return [f[:-len("_ts_raw.txt")] for f in get_ts_filenames(num_subjects)]


def get_raw_timeseries(subject_ids):
    """
    Gets raw timeseries arrays for the given list of subjects.

    Args:
        subject_ids: List of subject IDs.

    Returns:
        List of timeseries. Rows in timeseries correspond to brain regions, 
        columns correspond to timeseries values.
    """

    timeseries = []
    for subject_id in subject_ids:
        f = os.path.join(data_timeseries, subject_id + '_ts_raw.txt')
        print("Reading timeseries file %s" % f)
        timeseries.append(np.loadtxt(f, delimiter=','))

    return timeseries


# TODO: include the argument for the kind of connectivity matrix (partial
# correlation, correlation, lasso,...)
# TODO: save: Indicates whether to save the connectivity matrix to a file.
# TODO: save_path: Indicates the path where to store the connectivity matrix.

def get_functional_connectivity(timeseries):
    """
    Derives the correlation matrix for the parcellated timeseries data.

    Args:
        timeseries: Parcellated timeseries of shape [number ROI, timepoints].

    Returns:
        The flattened lower triangle of the correlation matrix for the 
        parcellated timeseries data.
    """

    conn_measure = connectome.ConnectivityMeasure(
        kind='correlation',
        vectorize=True,
        discard_diagonal=True)
    connectivity = conn_measure.fit_transform([np.transpose(timeseries)])[0]

    return connectivity


# TODO: get cortical thickness and Euler indices.

def get_structural_data(subject_ids):
    """
    Retrieves the non-timeseries data for the list of subjects.

    Args:
        subject_ids: List of subject IDs.

    Returns:
        The matrix containing the structural attributes for the list of
        subjects, of shape (num_subjects, num_structural_attributes)
    """

    # TODO: just assume sorted. 

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
            

def construct_population_graph(size, save=False, save_dir=None):
    subject_ids = get_subject_ids(size)
    raw_timeseries = get_raw_timeseries(subject_ids)
    connectivities = [get_functional_connectivity(ts) for ts in raw_timeseries]

    edge_index = torch.tensor(
        construct_edge_list(subject_ids), 
        dtype=torch.long)
    
    # TODO: train and test split.

    population_graph = Data(x=connectivities, edge_index=edge_index, y=None)

    if save:
        torch.save(population_graph, os.path.join(save_dir, 'population_graph.pt'))

    return population_graph

def load_population_graph(graph_root):
    return torch.load(os.path.join(graph_root, 'population_graph.pt'))


# subject_ids = get_subject_ids(1)
# print(subject_ids)
# ts = get_raw_timeseries(subject_ids)
# conn = get_functional_connectivity(ts[0])
