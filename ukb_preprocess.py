import os
from functools import reduce

import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure

data_root = 'data'
data_timeseries = 'data/raw_ts'
data_phenotype = 'data/phenotype.csv'
data_ct = 'data/CT.csv'
data_sa = 'data/SA.csv'
data_vol = 'data/Vol.csv'
data_euler = 'data/Euler.csv'
data_computed_fcms = 'data/processed_ts'

# Exclude the following raw timeseries due to incorrect size.
EXCLUDED_UKB_IDS = ['UKB2203847', 'UKB2208238', 'UKB2697888']


def precompute_fcm(subject_id=None):
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
    gmv = pd.read_csv(data_vol, sep=',', quotechar='\"')
    gmv_ids = gmv['NewID'].to_numpy()

    # Euler index IDs.
    euler = pd.read_csv(data_euler, sep=',', quotechar='\"')
    euler_ids = euler['eid'].to_numpy()

    # Save intersection of the subject IDs present in all datasets.
    intersected_ids = sorted(reduce(np.intersect1d,
                                    (timeseries_ids, phenotype_ids, ct_ids, sa_ids, gmv_ids, euler_ids)))
    np.save(os.path.join(data_root, 'subject_ids'), intersected_ids)
    return intersected_ids


if __name__ == '__main__':
    # precompute_fcm()
    precompute_subject_ids()
    subject_ids = np.load(os.path.join(data_root, 'subject_ids.npy'), allow_pickle=True)

    # 17550 = count of functional connectivity files
    # 3 = excluded functional connectivity files due to mismatched formatting
    # 233 = number of missing patients in phenotype data
    assert len(subject_ids) == (17550 - 233 - 3)

