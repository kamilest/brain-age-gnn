import numpy as np
import os

from nilearn.connectome import ConnectivityMeasure

if __name__ == '__main__':
    data_timeseries = 'data/raw_ts'

    conn_measure = ConnectivityMeasure(
        kind='correlation',
        vectorize=True,
        discard_diagonal=True)

    ts_filenames = [f for f in sorted(os.listdir(data_timeseries))]
    l = len('_ts_raw.txt')

    for ts_file in ts_filenames:
        print("Processing %s" % ts_file)
        ts = np.loadtxt(os.path.join(data_timeseries, ts_file), delimiter=',')
        np.save(os.path.join('data/processed_ts', ts_file[:-l]), conn_measure.fit_transform([np.transpose(ts)])[0])
