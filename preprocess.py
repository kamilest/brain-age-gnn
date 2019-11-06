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
import os.path as osp

from nilearn import connectome
from sklearn.covariance import EmpiricalCovariance

import torch
from torch_geometric.data import Data, Dataset

# Data sources.
root_folder = \
  '/Users/kamilestankeviciute/Google Drive/Part II/Dissertation/brain-age-gnn'
timeseries_data_folder = osp.join(root_folder, 'data/TS')
fcm_data_folder = osp.join(root_folder, 'data/fcm')


class PopulationGraph(Dataset):
  def __init__(self, root, transform=None, pre_transform=None):
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
    # Download to `self.raw_dir`.
    pass

  def process(self):
    i = 0
    for raw_path in self.raw_paths:
      # Read data from `raw_path`.
      data = Data()

      if self.pre_filter is not None and not self.pre_filter(data):
        continue

      if self.pre_transform is not None:
        data = self.pre_transform(data)

      torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
      i += 1

  def get(self, idx):
    data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx))
    return data



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
    for f in sorted(os.listdir(timeseries_data_folder))]

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
    fl = osp.join(timeseries_data_folder, subject_id + '_ts_raw.txt')
    print("Reading timeseries file %s" %fl)
    timeseries.append(np.loadtxt(fl, delimiter=','))

  return timeseries

#TODO: include the argument for the kind of connectivity matrix (partial 
# correlation, correlation, lasso,...)
#TODO: save: Indicates whether to save the connectivity matrix to a file.
#TODO: save_path: Indicates the path where to store the connectivity matrix.

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


def construct_population_graph(subject_ids):
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

  # for l in scores:
  #     label_dict = get_subject_score(subject_list, l)

  #     # quantitative phenotypic scores
  #     if l in ['AGE_AT_SCAN', 'FIQ']:
  #         for k in range(num_nodes):
  #             for j in range(k + 1, num_nodes):
  #                 try:
  #                     val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
  #                     if val < 2:
  #                         graph[k, j] += 1
  #                         graph[j, k] += 1
  #                 except ValueError:  # missing label
  #                     pass

  #     else:
  #         for k in range(num_nodes):
  #             for j in range(k + 1, num_nodes):
  #                 if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
  #                     graph[k, j] += 1
  #                     graph[j, k] += 1

  # return graph


subject_ids = get_subject_ids(1)
print(subject_ids)
ts = get_raw_timeseries(subject_ids)
conn = get_functional_connectivity(ts[0])
