import numpy as np
import pandas as pd
import os

from nilearn.connectome import ConnectivityMeasure

data_timeseries = 'data/raw_ts'
data_phenotype = 'data/phenotype.csv'
data_computed_fcms = 'data/preprocessed_ts'


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


# extract_phenotype_uids(['31-0.0', '21003-2.0'], ['UKB1000028', 'UKB1000133'])
def extract_phenotypes(uid_list, subject_ids):
    phenotype = pd.read_csv(data_phenotype, sep=',')
    subject_ids_no_UKB = [i[3:] for i in subject_ids]

    # Extract data for relevant subject IDs.
    subject_phenotype = phenotype[phenotype['eid'].isin(subject_ids_no_UKB)]

    # Extract relevant UIDs.
    subject_phenotype = subject_phenotype[uid_list]
    subject_phenotype.index = subject_ids

    return subject_phenotype


if __name__ == '__main__':
    precompute_fcm()
