import os
from functools import reduce

import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure

data_root = 'data'
data_timeseries = 'data/raw_ts'
data_phenotype = 'data/phenotype.csv'
data_ct = 'data/CT.csv'
data_euler = 'data/Euler.csv'
data_computed_fcms = 'data/processed_ts'

# Exclude the following raw timeseries due to incorrect size.
EXCLUDED_UKB_IDS = ['UKB2203847', 'UKB2208238', 'UKB2697888']


def precompute_fcm(subject_id=None):
    """
    Derives the correlation matrices for the parcellated timeseries data.

    Args:
        subject_id: Optional, if only one connectivity matrix needs to be precomputed.

    Returns:
        The flattened lower triangle of the correlation matrices for the parcellated timeseries data. Saved as a binary
        numpy array with the name of patient ID in the preprocessed timeseries directory.
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

    # Euler index IDs.
    euler = pd.read_csv(data_euler, sep=',', quotechar='\"')
    euler_ids = euler['eid'].to_numpy()

    # Save intersection of the subject IDs present in all datasets.
    intersected_ids = reduce(np.intersect1d, (timeseries_ids, phenotype_ids, ct_ids, euler_ids))
    np.save(os.path.join(data_root, 'subject_ids'), intersected_ids)


def precompute_similarity_feautres():
    """Precomputes the columns of the phenotype dataset for faster subject comparison.
    Saves the copy of the precomputed features in the phenotype dataset.

    :return: Saves the modified dataset under phenotype_precomputed.csv.
    """
    pass


# extract_phenotypes(['31-0.0', '21003-2.0'], ['UKB1000028', 'UKB1000133'])
def extract_phenotypes(subject_ids, uid_list=None):
    if uid_list is None:
        uid_list = ['eid']
    else:
        uid_list.append('eid')
    phenotype = pd.read_csv(data_phenotype, sep=',')
    subject_ids_no_UKB = [int(i[3:]) for i in subject_ids]

    # Extract data for relevant subject IDs.
    subject_phenotype = phenotype[phenotype['eid'].isin(subject_ids_no_UKB)]

    if len(subject_phenotype) != len(subject_ids):
        print('{} entries had phenotypic data missing.'.format(len(subject_ids) - len(subject_phenotype)))

    # Extract relevant UIDs.
    if len(uid_list) > 1:
        subject_phenotype = subject_phenotype[uid_list]

    # Add UKB prefix back to the index.
    subject_phenotype.index = ['UKB' + str(eid) for eid in subject_phenotype['eid']]
    subject_phenotype.sort_index()

    return subject_phenotype


def extract_cortical_thickness(subject_ids):
    ct = pd.read_csv(data_ct, sep=',', quotechar='\"')

    # Extract data for relevant subject IDs.
    subject_ct = ct[ct['NewID'].isin(subject_ids)]

    assert(len(subject_ids) - len(subject_ct) == 0)
    if len(subject_ct) != len(subject_ids):
        print('{} entries had cortical thickness data missing.'.format(len(subject_ids) - len(subject_ct)))

    subject_ct = subject_ct.drop(subject_ct.columns[0], axis=1)
    subject_ct = subject_ct.drop(['lh_???', 'rh_???'], axis=1)

    subject_ct.index = subject_ct['NewID']
    subject_ct = subject_ct.drop(['NewID'], axis=1)
    subject_ct = subject_ct.sort_index()

    return subject_ct


def extract_euler(subject_ids):
    euler = pd.read_csv(data_euler, sep=',', quotechar='\"')

    # Extract data for relevant subject IDs.
    subject_euler = euler[euler['eid'].isin(subject_ids)]
    assert (len(subject_ids) - len(subject_euler) == 0)

    subject_euler.index = subject_euler['eid']
    subject_euler = subject_euler.drop(['eid', 'oldID'], axis=1)
    subject_euler = subject_euler.sort_index()

    return subject_euler


if __name__ == '__main__':
    # precompute_fcm()
    precompute_subject_ids()
    subject_ids = np.load(os.path.join(data_root, 'subject_ids.npy'), allow_pickle=True)

    # 17550 = count of functional connectivity files
    # 3 = excluded functional connectivity files due to mismatched formatting
    # 233 = number of missing patients in phenotype data
    assert len(subject_ids) == (17550 - 233 - 3)
