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


def run_multiprocessing(func, i, n_processors):
    with Pool(processes=n_processors) as pool:
        return np.array(pool.map(func, i)).flatten()


def precompute_mental_similarities(n):
    subject_ids = np.load(SUBJECT_IDS, allow_pickle=True)
    similarity_lookup = pd.read_pickle(SIMILARITY_LOOKUP)

    sm = []
    mental_feature_codes = [Phenotype.MENTAL_HEALTH.value + str(i) for i in range(1, 19)]
    for i in range(2**n+1, min(2**(n+1)+1, 130)):
        id_i = subject_ids[i]
        for j in range(i+1, min(2**(n+1)+1, 130)):
            id_j = subject_ids[j]
            sm.append([i, j, np.dot(similarity_lookup.loc[id_i, mental_feature_codes],
                                    similarity_lookup.loc[id_j, mental_feature_codes]) >= 1])
    return np.array(sm)


def main():
    start = time.clock()

    num_processes = 14  # log_2(17314)
    thread_id = list(range(num_processes))

    out = run_multiprocessing(precompute_mental_similarities, thread_id, num_processes)
    np.save(os.path.join(similarity_root, 'MEN_multiprocess_similarity'), np.array(out))
    print('Total time: {}'.format(time.clock()-start))


if __name__ == "__main__":
    freeze_support()
    main()
