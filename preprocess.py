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

# from sklearn.feature_selection import RFE
# from nilearn import connectome

# Input data.
root_folder = \
  '/Users/kamilestankeviciute/Google Drive/Part II/Dissertation/brain-age-gnn'
data_folder = os.path.join(root_folder, 'data')


# Get timeseries arrays for list of subjects
def get_timeseries(subject_list, atlas_name):
  """
      subject_list : list of short subject IDs in string format
      atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200

  returns:
      time_series  : list of timeseries arrays, each of shape (timepoints x regions)
  """
  pass


# Compute connectivity matrices
def subject_connectivity(timeseries, subject, atlas_name, kind, save=True, save_path=data_folder):
  """
      timeseries   : timeseries table for subject (timepoints x regions)
      subject      : the subject ID
      atlas_name   : name of the parcellation atlas used
      kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
      save         : save the connectivity matrix to a file
      save_path    : specify path to save the matrix if different from subject folder

  returns:
      connectivity : connectivity matrix (regions x regions)
  """

  pass


# Get the list of subject IDs
def get_ids(num_subjects=None):
  """

  return:
      subject_IDs    : list of all subject IDs
  """

  subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject_IDs.txt'), dtype=str)

  if num_subjects is not None:
      subject_IDs = subject_IDs[:num_subjects]

  return subject_IDs


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
