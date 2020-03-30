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

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import ukb_preprocess
from phenotype import Phenotype

# Data sources.
data_root = 'data'
graph_root = 'data/graph'

data_timeseries = 'data/raw_ts'
data_phenotype = 'data/phenotype.csv'
data_similarity = 'data/similarity'
data_ct = 'data/CT.csv'
data_sa = 'data/SA.csv'
data_vol = 'data/Vol.csv'
data_euler = 'data/Euler.csv'
data_computed_fcms = 'data/processed_ts'

SUBJECT_IDS = 'data/subject_ids.npy'

# Exclude the following raw timeseries due to incorrect size.
EXCLUDED_UKB_IDS = ['UKB2203847', 'UKB2208238', 'UKB2697888']

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
        ukb_preprocess.precompute_subject_ids()

    subject_ids = np.load(os.path.join(data_root, 'subject_ids.npy'), allow_pickle=True)

    if not num_subjects:
        return subject_ids

    if randomise:
        np.random.seed(seed)
        return np.random.choice(subject_ids, num_subjects, replace=False)
    else:
        return subject_ids[:num_subjects]


def get_functional_connectivity(subject_id):
    """Returns the correlation matrix for the parcellated timeseries data.
    If necessary precomputes the matrix.

    :param subject_id: subject ID.
    :return: the flattened lower triangle of the correlation matrix for the parcellated timeseries data.
    """

    if subject_id + '.npy' not in os.listdir(data_computed_fcms):
        ukb_preprocess.precompute_fcm(subject_id)

    return np.load(os.path.join(data_computed_fcms, subject_id + '.npy'))


def get_all_functional_connectivities(subject_ids):
    connectivities = []
    for i, subject_id in enumerate(subject_ids):
        connectivity = get_functional_connectivity(subject_id)
        assert len(connectivity) == 70500
        connectivities.append(connectivity)

    return connectivities


def get_graph_name(size, functional, pca, structural, euler, similarity_feature_set, similarity_threshold):
    separator = '_'
    similarity_feature_string = separator.join([feature.value for feature in similarity_feature_set])
    return 'population_graph_' \
           + (str(size) + '_' if size is not None else 'all_') \
           + similarity_feature_string \
           + '_{}'.format(similarity_threshold) \
           + ('_functional' if functional else '') \
           + ('_PCA' if functional and pca else '') \
           + ('_structural' if structural else '') \
           + ('_euler' if euler else '') \
           + '.pt'


def collect_graph_data(subject_ids, functional, structural, euler):
    phenotypes = extract_phenotypes(subject_ids)
    assert len(np.intersect1d(subject_ids, phenotypes.index)) == len(subject_ids)

    if functional:
        functional_data = get_all_functional_connectivities(subject_ids)
    else:
        functional_data = pd.DataFrame(np.empty((len(subject_ids), 0)))

    if structural:
        cortical_thickness_data = extract_structural(subject_ids, type='cortical_thickness')
        assert len(np.intersect1d(subject_ids, cortical_thickness_data.index)) == len(subject_ids)

        surface_area_data = extract_structural(subject_ids, type='surface_area')
        assert len(np.intersect1d(subject_ids, surface_area_data.index)) == len(subject_ids)

        volume_data = extract_structural(subject_ids, type='volume')
        assert len(np.intersect1d(subject_ids, volume_data.index)) == len(subject_ids)
    else:
        cortical_thickness_data = pd.DataFrame(np.empty((len(subject_ids), 0)))
        surface_area_data = pd.DataFrame(np.empty((len(subject_ids), 0)))
        volume_data = pd.DataFrame(np.empty((len(subject_ids), 0)))

    if euler:
        euler_data = extract_euler(subject_ids)
        assert len(np.intersect1d(subject_ids, euler_data.index)) == len(subject_ids)
    else:
        euler_data = pd.DataFrame(pd.np.empty((len(subject_ids), 0)))

    return sorted(subject_ids), {'phenotypes': phenotypes,
                                 'functional': functional_data,
                                 'cortical_thickness': cortical_thickness_data,
                                 'surface_area': surface_area_data,
                                 'volume': volume_data,
                                 'euler': euler_data}


def get_sufficient_age_occurrence_index(phenotypes):
    age_counts = phenotypes[AGE_UID].value_counts()
    ages = age_counts[age_counts >= 3].index.tolist()
    age_index = np.where(phenotypes[AGE_UID].isin(ages))[0]
    return age_index


def construct_edge_list_from_function(subject_ids, similarity_function, similarity_threshold=0.5, save=False,
                                      graph_name=None):
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


def construct_edge_list(subject_ids, phenotypes, similarity_threshold=0.5):
    # Get the similarity matrices for subject_ids and phenotype_features provided.
    # Add up the matrices (possibly weighting).
    num_subjects = len(subject_ids)
    similarities = np.zeros((num_subjects, num_subjects), dtype=np.float32)

    full_subject_ids = np.load(SUBJECT_IDS, allow_pickle=True)
    id_mask = np.isin(full_subject_ids, subject_ids)

    for ph in phenotypes:
        ph_similarity = np.load(os.path.join(data_similarity, '{}_similarity.npy'.format(ph.value)))
        similarities += ph_similarity[id_mask][:, id_mask].astype(np.float32)

    # Take the average
    similarities /= len(phenotypes)

    # Filter values above threshold
    return np.transpose(np.argwhere(similarities >= similarity_threshold))


def construct_population_graph(similarity_feature_set, similarity_threshold=0.5, size=None, functional=False,
                               pca=False, structural=True, euler=True, save=True, subject_ids=None, age_filtering=True,
                               save_dir=graph_root, name=None):
    if name is None:
        name = get_graph_name(size, functional, pca, structural, euler, similarity_feature_set, similarity_threshold)

    if subject_ids is None:
        subject_ids = sorted(get_subject_ids(size))

    # Collect the required data.
    subject_ids, graph_data = collect_graph_data(subject_ids, functional, structural, euler)

    # Remove subjects with too few instances of the label for stratification.
    if age_filtering:
        age_index = get_sufficient_age_occurrence_index(graph_data['phenotypes'])
        subject_ids = sorted(graph_data['phenotypes'].iloc[age_index].index.tolist())
        for feature in graph_data.keys():
            if graph_data[feature] is not None:
                graph_data[feature] = graph_data[feature].iloc[age_index].copy()

    num_subjects = len(subject_ids)
    print('{} subjects remaining for population_graph construction.'.format(num_subjects))

    labels = graph_data['phenotypes'].loc[subject_ids, AGE_UID].to_numpy()
    label_tensor = torch.tensor([labels], dtype=torch.float32).transpose_(0, 1)

    # Construct the edge index.
    edge_index_tensor = torch.tensor(
        construct_edge_list(subject_ids=subject_ids, phenotypes=similarity_feature_set,
                            similarity_threshold=similarity_threshold),
        dtype=torch.long)
    # similarity_function = similarity.custom_similarity_function(similarity_feature_set)
    # construct_edge_list_from_function(subject_ids=subject_ids, similarity_function=similarity_function,
    # similarity_threshold=similarity_threshold, save=logs, graph_name=name.replace('.pt', datetime.now(
    # ).strftime("_%H_%M_%S") + '.csv')), dtype=torch.long)

    population_graph = Data()
    population_graph.num_nodes = len(subject_ids)
    population_graph.subject_index = subject_ids

    population_graph.edge_index = edge_index_tensor
    population_graph.y = label_tensor

    population_graph.functional_data = graph_data['functional']
    population_graph.structural_data = {'cortical_thickness': graph_data['cortical_thickness'],
                                        'surface_area': graph_data['surface_area'],
                                        'volume': graph_data['volume']}
    population_graph.euler_data = graph_data['euler']
    population_graph.name = name

    if save:
        torch.save(population_graph, os.path.join(save_dir, name))

    return population_graph


def load_population_graph(graph_root, name):
    return torch.load(os.path.join(graph_root, name))


if __name__ == '__main__':
    feature_set = [Phenotype.SEX, Phenotype.MENTAL_HEALTH, Phenotype.PROSPECTIVE_MEMORY_RESULT, Phenotype.BIPOLAR_DISORDER_STATUS, Phenotype.NEUROTICISM_SCORE]
    graph = construct_population_graph(feature_set, similarity_threshold=0.8)
    # graph = load_population_graph(graph_root, 'population_graph_all_SEX_FTE_FI_structural_euler.pt')


def extract_phenotypes(subject_ids, uid_list=None):
    if uid_list is None:
        uid_list = ['eid']
    else:
        uid_list.append('eid')
    phenotype = pd.read_csv(data_phenotype, sep=',')
    subject_ids_no_UKB = [int(i[3:]) for i in subject_ids]

    # Extract data for relevant subject IDs.
    subject_phenotype = phenotype[phenotype['eid'].isin(subject_ids_no_UKB)].copy()

    if len(subject_phenotype) != len(subject_ids):
        print('{} entries had phenotypic data missing.'.format(len(subject_ids) - len(subject_phenotype)))

    # Extract relevant UIDs.
    if len(uid_list) > 1:
        subject_phenotype = subject_phenotype[uid_list]

    # Add UKB prefix back to the index.
    subject_phenotype.index = ['UKB' + str(eid) for eid in subject_phenotype['eid']]
    subject_phenotype.sort_index()

    return subject_phenotype


def extract_structural(subject_ids, type):
    if type == 'cortical_thickness':
        data = data_ct
    elif type == 'surface_area':
        data = data_sa
    elif type == 'volume':
        data = data_vol
    else:
        return pd.DataFrame(pd.np.empty((len(subject_ids), 0)))

    ct = pd.read_csv(data, sep=',', quotechar='\"')

    # Extract data for relevant subject IDs.
    subject_ct = ct[ct['NewID'].isin(subject_ids)].copy()

    assert(len(subject_ids) - len(subject_ct) == 0)
    if len(subject_ct) != len(subject_ids):
        print('{} entries had {} data missing.'.format(len(subject_ids) - len(subject_ct), type))

    subject_ct = subject_ct.drop(subject_ct.columns[0], axis=1)
    subject_ct = subject_ct.drop(['lh_???', 'rh_???'], axis=1)

    subject_ct.index = subject_ct['NewID']
    subject_ct = subject_ct.drop(['NewID'], axis=1)
    subject_ct = subject_ct.sort_index()

    return subject_ct


def extract_euler(subject_ids):
    euler = pd.read_csv(data_euler, sep=',', quotechar='\"')

    # Extract data for relevant subject IDs.
    subject_euler = euler[euler['eid'].isin(subject_ids)].copy()
    assert (len(subject_ids) - len(subject_euler) == 0)

    subject_euler.index = subject_euler['eid']
    subject_euler = subject_euler.drop(['eid', 'oldID'], axis=1)
    subject_euler = subject_euler.sort_index()

    return subject_euler
