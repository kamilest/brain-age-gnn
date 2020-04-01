"""
Population graph construction component.

Collects the specified population graph modalities for a given number of subjects (or their specific IDs).
Takes a set of similarity metrics (or a generalised similarity function) to generate the edges based on subject
    similarity.
Combines the imaging data and the edges into the intermediate population graph representation.
"""

import csv
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import ukb_preprocess
from phenotype import Phenotype
from ukb_preprocess import ICD10_LOOKUP

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

    :param num_subjects: number of subjects. Use the entire dataset when set to None.
    :param randomise: indicates whether to use a random seed for selection of subjects.
    :param seed: random seed value.
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
    """Returns the flattened functional connectivity matrix for the parcellated timeseries data of a given subject.
    Precomputes the functional connectivity matrix if necessary.

    :param subject_id: subject ID.
    :return: the flattened lower triangle of the correlation matrix for the parcellated timeseries data.
    """

    if subject_id + '.npy' not in os.listdir(data_computed_fcms):
        ukb_preprocess.precompute_flattened_fcm(subject_id)

    return np.load(os.path.join(data_computed_fcms, subject_id + '.npy'))


def collect_functional_connectivities(subject_ids):
    """Returns a list of flattened functional connectivity matrices for a list of subjects.
    Precomputes the functional connectivity matrices if necessary.

    :param subject_ids: list of subject IDs.
    :return: the flattened lower triangles of the correlation matrices for the parcellated timeseries data.
    """

    connectivities = []
    for i, subject_id in enumerate(subject_ids):
        connectivity = get_functional_connectivity(subject_id)
        assert len(connectivity) == 70500
        connectivities.append(connectivity)

    return connectivities


def collect_phenotypes(subject_ids, uid_list=None):
    """Returns the phenotype data for a given list of subjects and the list of phenotype identifiers.

    :param subject_ids: list of subject IDs.
    :param uid_list: list of the phenotype data identifiers. If list is None return all phenotype data.
    :return: a dataframe indexed by sorted subject IDs with columns containing the phenotype data.
    """

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


def collect_structural(subject_ids, structural_type):
    """Returns the structural data for a given list of subjects and the specified structural data type.

    :param subject_ids: list of subject IDs.
    :param structural_type: indicates whether to retrieve 'cortical_thickness', 'surface_area',
        or (grey matter) 'volume'.
    :return: a dataframe indexed by sorted subject IDs with columns containing the structural data.
    """

    if structural_type == 'cortical_thickness':
        data = data_ct
    elif structural_type == 'surface_area':
        data = data_sa
    elif structural_type == 'volume':
        data = data_vol
    else:
        return pd.DataFrame(pd.np.empty((len(subject_ids), 0)))

    ct = pd.read_csv(data, sep=',', quotechar='\"')

    # Extract data for relevant subject IDs.
    subject_ct = ct[ct['NewID'].isin(subject_ids)].copy()

    assert(len(subject_ids) - len(subject_ct) == 0)
    if len(subject_ct) != len(subject_ids):
        print('{} entries had {} data missing.'.format(len(subject_ids) - len(subject_ct), structural_type))

    subject_ct = subject_ct.drop(subject_ct.columns[0], axis=1)
    subject_ct = subject_ct.drop(['lh_???', 'rh_???'], axis=1)

    subject_ct.index = subject_ct['NewID']
    subject_ct = subject_ct.drop(['NewID'], axis=1)
    subject_ct = subject_ct.sort_index()

    return subject_ct


def collect_euler(subject_ids):
    """Returns the Euler index data for a given list of subjects.

    :param subject_ids: list of subject IDs.
    :return: a dataframe indexed by sorted subject IDs with columns containing the Euler indices.
    """

    euler = pd.read_csv(data_euler, sep=',', quotechar='\"')

    # Extract data for relevant subject IDs.
    subject_euler = euler[euler['eid'].isin(subject_ids)].copy()
    assert (len(subject_ids) - len(subject_euler) == 0)

    subject_euler.index = subject_euler['eid']
    subject_euler = subject_euler.drop(['eid', 'oldID'], axis=1)
    subject_euler = subject_euler.sort_index()

    return subject_euler


def collect_graph_data(subject_ids, functional, structural, euler):
    """Collects the graph data as for the given subject IDs and the specified list of modalities.

    :param subject_ids: list of subject IDs.
    :param functional: indicates whether to use functional MRI data.
    :param structural: indicates whether to use structural MRI data.
    :param euler: indicates whether to use Euler indices.
    :return: a dictionary containing the dataframes for every modality, indexed by subject IDs. If the modality is not
        used, an empty dataframe is returned in the corresponding dictionary entry.
    """

    phenotypes = collect_phenotypes(subject_ids)
    assert len(np.intersect1d(subject_ids, phenotypes.index)) == len(subject_ids)

    if functional:
        functional_data = collect_functional_connectivities(subject_ids)
    else:
        functional_data = pd.DataFrame(np.empty((len(subject_ids), 0)))

    if structural:
        cortical_thickness_data = collect_structural(subject_ids, structural_type='cortical_thickness')
        assert len(np.intersect1d(subject_ids, cortical_thickness_data.index)) == len(subject_ids)

        surface_area_data = collect_structural(subject_ids, structural_type='surface_area')
        assert len(np.intersect1d(subject_ids, surface_area_data.index)) == len(subject_ids)

        volume_data = collect_structural(subject_ids, structural_type='volume')
        assert len(np.intersect1d(subject_ids, volume_data.index)) == len(subject_ids)
    else:
        cortical_thickness_data = pd.DataFrame(np.empty((len(subject_ids), 0)))
        surface_area_data = pd.DataFrame(np.empty((len(subject_ids), 0)))
        volume_data = pd.DataFrame(np.empty((len(subject_ids), 0)))

    if euler:
        euler_data = collect_euler(subject_ids)
        assert len(np.intersect1d(subject_ids, euler_data.index)) == len(subject_ids)
    else:
        euler_data = pd.DataFrame(pd.np.empty((len(subject_ids), 0)))

    return sorted(subject_ids), {'phenotypes': phenotypes,
                                 'functional': functional_data,
                                 'cortical_thickness': cortical_thickness_data,
                                 'surface_area': surface_area_data,
                                 'volume': volume_data,
                                 'euler': euler_data}


def get_graph_name(size, functional, structural, euler, similarity_feature_set, similarity_threshold):
    """Creates the name for the population graph given its parameterisation.

    :param size: list of subject IDs.
    :param functional: indicates whether the graph contains functional MRI data.
    :param structural: indicates whether the graph contains structural MRI data.
    :param euler: indicates whether the graph contains Euler indices.
    :param similarity_feature_set: list of similarity features were used in similarity metric.
    :param similarity_threshold: the threshold of similary metric above which an edge is created.
    :return: the graph name.
    """

    similarity_feature_string = '_'.join(sorted([feature.value for feature in similarity_feature_set]))
    return 'population_graph_' \
           + (str(size) + '_' if size is not None else '_') \
           + similarity_feature_string \
           + '_{}'.format(similarity_threshold) \
           + ('_functional' if functional else '') \
           + ('_structural' if structural else '') \
           + ('_euler' if euler else '') \
           + '.pt'


def get_sufficient_age_occurrence_index(phenotypes):
    """Required for correct stratification; ensures that occurrence of each label is at least 3.

    :param phenotypes: dataframe containing phenotype (non-imaging data).
    :return: return subject IDs whose ages occur with sufficient frequency to be correctly stratified into training,
        validation and test sets.
    """

    age_counts = phenotypes[AGE_UID].value_counts()
    ages = age_counts[age_counts >= 3].index.tolist()
    age_index = np.where(phenotypes[AGE_UID].isin(ages))[0]
    return age_index


def get_healthy_brain_subject_mask(subject_ids):
    """Returns the boolean mask for subjects which have healthy brains.
    Brain health is defined as having no mental health or nervous system disorder diagnosis.

    :param subject_ids: list of subject IDs.
    :return boolean mask indicating brain health.
    """

    full_subject_ids = np.load(SUBJECT_IDS, allow_pickle=True)
    si = np.searchsorted(full_subject_ids, subject_ids)

    icd10_lookup = pd.read_pickle(ICD10_LOOKUP).to_numpy()
    return np.invert(np.any(icd10_lookup[si], axis=1))


def construct_edge_list_from_function(subject_ids, similarity_function, similarity_threshold=0.5, save=False,
                                      graph_name=None):
    """Constructs the adjacency list of the population population_graph based on a similarity metric provided.

    :param subject_ids: subject IDs.
    :param similarity_function: function which is returns similarity between two subjects according to some metric.
    :param similarity_threshold: the threshold above which the edge should be added.
    :param save: inidicates whether to save the population_graph in the logs directory.
    :param graph_name: population_graph name for saved file if population_graph edges are logged.
    :return population_graph connectivity in coordinate format of shape [2, num_edges].
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
    """Constructs the adjacency list of the population population_graph based on a similarity features provided.

    :param subject_ids: list of subject IDs.
    :param phenotypes: the list of features which will be accounted for with equal importance when computing similarity.
    :param similarity_threshold: the threshold above which the edge will be added to the graph.
    :return a dictionary containing the dataframes for every modality, indexed by subject IDs. If the modality is not
        used, an empty dataframe is returned in the corresponding dictionary entry.
    """

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
                               structural=True, euler=True, save=True, subject_ids=None, age_filtering=True,
                               save_dir=graph_root, name=None):
    """Constructs the population graph given its modality and similarity parameterisation.

    :param similarity_feature_set: list of features which will be accounted for with equal importance when computing
        similarity.
    :param similarity_threshold: the threshold above which the edge will be added to the graph.
    :param size: the maximum number of nodes in population graph (can be lower if some nodes have to be filtered out).
        If size is None then the entire dataset is considered.
    :param functional: indicates whether to use functional MRI data.
    :param structural: indicates whether to use structural MRI data.
    :param euler: indicates whether to use Euler index data.
    :param save: indicates whether the graph should be saved.
    :param subject_ids: the specific subjects for which population graph should be created. Takes precedence over the
        size parameter if both are indicated.
    :param age_filtering: indicates whether to remove patients with low label occurrence (for correct stratification in
        training process).
    :param save_dir: directory in which the graph should be saved.
    :param name: the name of the graph. Creates a default name if None is given.
    :return an intermediate population graph representation.
    """

    if name is None:
        name = get_graph_name(size, functional, structural, euler, similarity_feature_set, similarity_threshold)

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
    population_graph.healthy_brain_subject_mask = get_healthy_brain_subject_mask(subject_ids)

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


def load_population_graph(root, name):
    """Loads population graph.

    :param root: the root directory of the graph.
    :param name: the name of the population graph file.
    :return population graph object.
    """

    return torch.load(os.path.join(root, name))


if __name__ == '__main__':
    feature_set = [Phenotype.SEX, Phenotype.MENTAL_HEALTH, Phenotype.PROSPECTIVE_MEMORY_RESULT,
                   Phenotype.BIPOLAR_DISORDER_STATUS, Phenotype.NEUROTICISM_SCORE]
    graph = construct_population_graph(feature_set, similarity_threshold=0.8)
