import numpy as np
import os

from nilearn.connectome import ConnectivityMeasure

data_timeseries = 'data/raw_ts'
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


if __name__ == '__main__':
    precompute_fcm()
