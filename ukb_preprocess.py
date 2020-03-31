import os
from functools import reduce

import numpy as np
import pandas as pd
import torch
from nilearn.connectome import ConnectivityMeasure

from phenotype import Phenotype

data_root = 'data'
data_timeseries = 'data/raw_ts'
data_phenotype = 'data/phenotype.csv'
data_similarity = 'data/similarity'
data_ct = 'data/CT.csv'
data_sa = 'data/SA.csv'
data_gmv = 'data/Vol.csv'
data_euler = 'data/Euler.csv'
data_icd10 = 'data/ICD10.csv'
data_computed_fcms = 'data/processed_ts'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

SIMILARITY_LOOKUP = 'data/similarity_lookup.pkl'
SUBJECT_IDS = 'data/subject_ids.npy'

# Exclude the following raw timeseries due to incorrect size.
EXCLUDED_UKB_IDS = ['UKB2203847', 'UKB2208238', 'UKB2697888']


def precompute_flattened_fcm(subject_id=None):
    """ Derives the correlation matrices for the parcellated timeseries data.

    :param subject_id: subject ID if only one connectivity matrix needs to be precomputed
    :return: the flattened lower triangle of the correlation matrices for the parcellated timeseries data.
    Saved as a binary numpy array with the name of patient ID in the preprocessed timeseries directory.
    """

    conn_measure = ConnectivityMeasure(
        kind='correlation',
        vectorize=True,
        discard_diagonal=True)

    suffix = '_ts_raw.txt'

    if subject_id is not None:
        print("Processing %s" % subject_id)
        ts = np.loadtxt(os.path.join(data_timeseries, subject_id + suffix), delimiter=',')
        np.save(os.path.join(data_computed_fcms, subject_id), conn_measure.fit_transform([np.transpose(ts)])[0])

    else:
        # Preompute all timeseries.
        ts_filenames = [f for f in os.listdir(data_timeseries)]
        suffix_length = len(suffix)

        for ts_file in ts_filenames:
            print("Processing %s" % ts_file)
            ts = np.loadtxt(os.path.join(data_timeseries, ts_file), delimiter=',')
            np.save(os.path.join(data_computed_fcms,
                                 ts_file[:-suffix_length]),
                    conn_measure.fit_transform([np.transpose(ts)])[0])


def precompute_subject_ids():
    """Precomputes the index of subjects that have data available for all possible modalities.

    :return list of filtered subject IDs.
    """

    # Timeseries subject IDs.
    timeseries_ids = [f[:-len("_ts_raw.txt")] for f in sorted(os.listdir(data_timeseries))]

    for i in EXCLUDED_UKB_IDS:
        timeseries_ids.remove(i)

    # Phenotype data IDs.
    phenotype = pd.read_csv(data_phenotype, sep=',')
    phenotype_ids = np.array(['UKB' + str(eid) for eid in phenotype['eid']])

    # Cortical thickness IDs.
    ct = pd.read_csv(data_ct, sep=',', quotechar='\"')
    ct_ids = ct['NewID'].to_numpy()

    # Surface area IDs.
    sa = pd.read_csv(data_sa, sep=',', quotechar='\"')
    sa_ids = sa['NewID'].to_numpy()

    # Grey matter volume IDs.
    gmv = pd.read_csv(data_gmv, sep=',', quotechar='\"')
    gmv_ids = gmv['NewID'].to_numpy()

    # Euler index IDs.
    euler = pd.read_csv(data_euler, sep=',', quotechar='\"')
    euler_ids = euler['eid'].to_numpy()

    # ICD10 index IDs.
    icd10 = pd.read_csv(data_icd10, sep=',', quotechar='\"')
    icd10_ids = np.array(['UKB' + str(eid) for eid in icd10['eid']])

    # Save intersection of the subject IDs present in all datasets.
    intersected_ids = sorted(reduce(np.intersect1d,
                                    (timeseries_ids, phenotype_ids, ct_ids, sa_ids, gmv_ids, euler_ids, icd10_ids)))
    np.save(os.path.join(data_root, 'subject_ids'), intersected_ids)
    return intersected_ids


def get_most_recent(ukb_feature, subject_id, phenotypes):
    """Utility method for retrieving the most recent datapoint if several measurements of the same feature are
        available.

    :param ukb_feature: the UKB feature as a list of relevant UKB IDs.
    :param subject_id: the subject for which to return the most recent feature instance.
    :param phenotypes: the dataframe containing phenotype data.
    :return the value for the most recent feature.
    """

    instance = ukb_feature[0]
    for f in reversed(ukb_feature):
        if not np.isnan(phenotypes.loc[subject_id, f]):
            instance = f
            break
    return phenotypes.loc[subject_id, instance]


def create_similarity_lookup():
    """Precomputes the columns of the phenotype dataset for faster subject comparison.

    :return: dataframe containing the values used for similarity comparison, row-indexed by subject ID and
    column-indexed by phenotype code name (e.g. 'AGE', 'FTE' etc.)
    """

    phenotypes = pd.read_csv(data_phenotype, sep=',')
    phenotypes.index = ['UKB' + str(eid) for eid in phenotypes['eid']]

    biobank_feature_list = []
    for feature in Phenotype:
        biobank_feature_list.extend(Phenotype.get_biobank_codes(feature))

    phenotype_processed = phenotypes[biobank_feature_list]

    for feature in Phenotype:
        biobank_feature = Phenotype.get_biobank_codes(feature)
        if feature == Phenotype.MENTAL_HEALTH:
            mental_to_code = Phenotype.get_mental_to_code()
            # column names for summary (total number of conditions) + 18 possible condidions: MEN0, MEN1, ..., MEN18.
            mental_feature_codes = [Phenotype.MENTAL_HEALTH.value + str(i) for i in range(19)]
            # Replace string descriptions with their codes for consistency.
            phenotype_processed.loc[:, biobank_feature[0]] = phenotype_processed[biobank_feature[0]].apply(
                lambda x: mental_to_code[x] if x in mental_to_code.keys() else None)
            # Determine if the the patient has the occurrence of a particular disease.
            si = phenotype_processed.index.to_series()
            for i in range(1, len(mental_feature_codes)):
                phenotype_processed.loc[:, Phenotype.MENTAL_HEALTH.value + str(i)] = si.apply(
                    lambda s: int(i in phenotype_processed.loc[s, biobank_feature].to_numpy().astype(bool)))
            phenotype_processed.loc[:, mental_feature_codes[0]] = si.apply(
                lambda s: int(np.sum(phenotype_processed.loc[s, mental_feature_codes[1:]])))

        elif len(biobank_feature) > 1:
            # handle the more/less recent values
            si = phenotype_processed.index.to_series().copy()
            phenotype_processed.loc[:, feature.value] = si.apply(lambda s: get_most_recent(biobank_feature, s, phenotype_processed))
        else:
            phenotype_processed.loc[:, feature.value] = phenotype_processed[biobank_feature[0]].copy()

    # Filter only the subjects used in the final dataset.
    phenotype_processed = phenotype_processed.loc[precompute_subject_ids()]

    # Return only the final feature columns (indexed by code names).
    phenotype_processed.drop(biobank_feature_list, axis=1, inplace=True)
    phenotype_processed = phenotype_processed.sort_index()

    phenotype_processed.to_pickle(SIMILARITY_LOOKUP)
    return phenotype_processed


def precompute_similarities():
    """Creates the similarity metric based on the phenotype feature list.
    If a feature has several entries in the UK Biobank, take either the most recent available estimate or, if the
    entries correspond to categories, consider the matching category values.

    The final score is an average of all the indicator scores for each feature, i.e. if two subjects have all of the
    features matching, the total score will be 1, and if none of the features match then the value will be 0. Edge
    creation then depends on the similarity threshold defined in graph construction.

    If both features are unknown, assume there is no match.
    """

    p_list = Phenotype
    subject_ids = np.load(SUBJECT_IDS, allow_pickle=True)
    similarity_lookup = pd.read_pickle(SIMILARITY_LOOKUP)

    for p in p_list:
        if p == Phenotype.MENTAL_HEALTH:
            mental_feature_codes = [Phenotype.MENTAL_HEALTH.value + str(i) for i in range(1, 19)]
            men = similarity_lookup.loc[subject_ids, mental_feature_codes].to_numpy()
            men = torch.tensor(men)

            sim = torch.mm(men, men.t())
            sim = sim >= 1
            sm = sim.cpu().detach().numpy()

        else:
            fea = similarity_lookup.loc[subject_ids, p.value].to_numpy()
            fea = np.expand_dims(fea, axis=0)
            fea = torch.tensor(fea)

            sim = fea.t() - fea
            sim = sim == 0
            sm = sim.cpu().detach().numpy()

        # Ignore self-similarities
        np.fill_diagonal(sm, False)
        np.save(os.path.join(data_similarity, '{}_similarity'.format(p.value)), sm)


if __name__ == '__main__':
    # precompute_flattened_fcm()
    precompute_subject_ids()
    # precompute_similarities()
    sids = np.load(os.path.join(data_root, 'subject_ids.npy'), allow_pickle=True)

    # 17550 = count of functional connectivity files
    # 3 = excluded functional connectivity files due to mismatched formatting
    # 233 = number of missing patients in phenotype data
    assert len(sids) == (17550 - 233 - 3)

