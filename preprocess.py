"""
Graph preprocessing file.
Collects features and assembles the population graph.


Collects the relevant timeseries, 
computes functional/structural connectivity matrices
computes graph adjacency scores
connects nodes into a graph, assigning collected features
"""

import os
import numpy as np
import scipy.io as sio

# Data sources.
root_folder = \
  '/Users/kamilestankeviciute/Google Drive/Part II/Dissertation/brain-age-gnn'
timeseries_data_folder = os.path.join(root_folder, 'data/TS')


def get_subject_ids(num_subjects=None):
  """
    Gets the list of subject IDs for a spcecified number of subjects.
    If the number of subjects is not specified, all IDs are returned.
  
  Args:
    num_subjects: The number of subjects.

  Returns:
    List of subject IDs.
  """

  subject_ids = [f[:-len("_ts_raw.txt")] \
    for f in sorted(os.listdir(timeseries_data_folder))]

  if num_subjects is not None:
    subject_ids = subject_ids[:num_subjects]

  return subject_ids

def get_raw_timeseries(subject_ids):
  """Gets raw timeseries arrays for the given list of subjects.

  Args:
    subject_ids: List of subject IDs.

  Returns:
    List of timeseries. Rows in timeseries correspond to brain regions, 
    columns correspond to timeseries values.
  """

  timeseries = []
  for subject_id in subject_ids:
    fl = os.path.join(timeseries_data_folder, subject_id + '_ts_raw.txt')
    print("Reading timeseries file %s" %fl)
    timeseries.append(np.loadtxt(fl, delimiter=','))

  return timeseries

#TODO: include the argument for the kind of connectivity matrix (partial correlation, correlation, lasso,...)
def get_functional_connectivity(timeseries, subject, kind, save=True, save_path=''):
  """Derive the correlation matrix for the parcellated timeseries data.

  Args:
    timeseries: The parcellated timeseries of shape (number ROI x timepoints).
    subject: Subject ID.
    save: Indicates whether to save the connectivity matrix to a file.
    save_path: Indicates the path where to store the connectivity matrix.

  Returns:
    The flattened lower triangle of the correlation matrix for the parcellated
    timeseries data.
  """
  

  pass


# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
  # scores_dict = {}

  # with open(phenotype) as csv_file:
  #     reader = csv.DictReader(csv_file)
  #     for row in reader:
  #         if row['SUB_ID'] in subject_list:
  #             scores_dict[row['SUB_ID']] = row[score]

  # return scores_dict

  pass


# Construct the adjacency matrix of the population from phenotypic scores
def create_affinity_graph_from_scores(scores, subject_list):
  """
      scores       : list of phenotypic information to be used to construct the affinity graph
      subject_list : list of subject IDs

  return:
      graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
  """
  pass

  # num_nodes = len(subject_list)
  # graph = np.zeros((num_nodes, num_nodes))

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
print(ts)
