"""
Graph preprocessing file.
Collects features and assembles the population graph.


Collects the relevant timeseries, 
computes functional/structural connectivity matrices
computes graph adjacency scores
connects nodes into a graph, assigning collected features
"""

import numpy as np
import scipy.io as sio
import os

from nilearn import connectome

import torch
from torch_geometric.data import Data, Dataset

# Data sources.
data_root = \
    '/Users/kamilestankeviciute/Google Drive/Part II/Dissertation/' \
    'brain-age-gnn/data'
data_timeseries = os.path.join(data_root, 'data/raw_ts')
graph_root = os.path.join(data_root, 'graph')


class PopulationGraph(Dataset):

    def __init__(self, root, size, transform=None, pre_transform=None, pre_filter=None):
        super(PopulationGraph, self).__init__(root, transform, pre_transform)
  def __init__(self, root, size, transform=None, pre_transform=None, pre_filter=None):
    super(PopulationGraph, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]

    def __len__(self):
        return len(self.processed_file_names)

    def download(self):
        pass

    def process(self):
        i = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data(x=None, edge_index=None, edge_attr=None, y=None, pos=None, norm=None, face=None)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


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

    subject_ids = [f[:-len("_ts_raw.txt")]
                   for f in sorted(os.listdir(data_timeseries))]

    if num_subjects is not None:
        subject_ids = subject_ids[:num_subjects]

    return subject_ids


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
        fl = os.path.join(data_timeseries, subject_id + '_ts_raw.txt')
        print("Reading timeseries file %s" % fl)
        timeseries.append(np.loadtxt(fl, delimiter=','))

    return timeseries


# TODO: include the argument for the kind of connectivity matrix (partial
# correlation, correlation, lasso,...)
# TODO: save: Indicates whether to save the connectivity matrix to a file.
# TODO: save_path: Indicates the path where to store the connectivity matrix.

def get_functional_connectivity(timeseries):
    """
  Derives the correlation matrix for the parcellated timeseries data.

  Args:
    timeseries: The parcellated timeseries of shape (number ROI x timepoints).
    subject_id: Subject ID.

  Returns:
    The flattened lower triangle of the correlation matrix for the parcellated
    timeseries data.
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
        ???
    """

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


def construct_edge_list(subject_ids):
    """
    Constructs the adjacency list of the population graph based on the
    similarity metric.
  
    Args:
        subject_ids: List of subject IDs.

    Returns:
        An adjacency list of the population graph of the form
        {index: [neighbour_nodes]}, indexed by Subject IDs.
    """

    pass


subject_ids = get_subject_ids(1)
print(subject_ids)
ts = get_raw_timeseries(subject_ids)
conn = get_functional_connectivity(ts[0])
