import os
import time
from multiprocessing import Pool
from multiprocessing import freeze_support

import numpy as np
import pandas as pd

from phenotype import Phenotype

data_root = 'data'
data_phenotype = 'data/phenotype.csv'
similarity_root = 'data/similarity'

SIMILARITY_LOOKUP = 'data/similarity_lookup.pkl'
SUBJECT_IDS = 'data/subject_ids.npy'

NUM_PROCESSES = 16

def run_multiprocessing(func, i, n_processors):
    with Pool(processes=n_processors) as pool:
        return np.concatenate(np.array(pool.map(func, i)).flatten())


def precompute_mental_similarities(n):
    subject_ids = np.load(SUBJECT_IDS, allow_pickle=True)
    similarity_lookup = pd.read_pickle(SIMILARITY_LOOKUP)

    sm = []
    mental_feature_codes = [Phenotype.MENTAL_HEALTH.value + str(i) for i in range(1, 19)]
    lo = int(len(subject_ids) / NUM_PROCESSES) * n
    hi = min(int(len(subject_ids) / NUM_PROCESSES) * (n+1), len(subject_ids))
    print('Processing {}-{}'.format(lo, hi))
    for i in range(lo, hi):
        print('Subject {}'.format(i))
        id_i = subject_ids[i]
        for j in range(i+1, len(subject_ids)):
            id_j = subject_ids[j]
            sm.append([i, j, np.any(np.logical_and(similarity_lookup.loc[id_i, mental_feature_codes],
                                                   similarity_lookup.loc[id_j, mental_feature_codes]))])
    print('Finished processing {}-{}'.format(lo, hi))
    return np.array(sm)


def main():
    start = time.clock()
    wall_start = time.time()

    num_processes = NUM_PROCESSES
    thread_id = list(range(num_processes+1))

    out = run_multiprocessing(precompute_mental_similarities, thread_id, num_processes)
    np.save(os.path.join(similarity_root, 'MEN_multiprocess_similarity_nemo'), np.array(out))
    print('Total time: {}'.format(time.clock()-start))
    print('Total wall time: {}'.format((time.time()-wall_start)))


if __name__ == "__main__":
    freeze_support()
    main()
    # a = np.load('data/similarity/MEN_multiprocess_similarity.npy', allow_pickle=True)
