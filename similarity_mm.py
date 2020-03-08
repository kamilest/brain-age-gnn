import os
import time

import numpy as np
import pandas as pd
import torch

from phenotype import Phenotype

data_root = 'data'
data_phenotype = 'data/phenotype.csv'
similarity_root = 'data/similarity'

SIMILARITY_LOOKUP = 'data/similarity_lookup.pkl'
SUBJECT_IDS = 'data/subject_ids.npy'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def precompute_mental_similarities():
    subject_ids = np.load(SUBJECT_IDS, allow_pickle=True)
    similarity_lookup = pd.read_pickle(SIMILARITY_LOOKUP)
    mental_feature_codes = [Phenotype.MENTAL_HEALTH.value + str(i) for i in range(1, 19)]
    men = similarity_lookup.loc[subject_ids, mental_feature_codes].to_numpy()
    men = torch.tensor(men)

    sim = torch.mm(men, men.t())
    sim = sim >= 1

    sim = sim.cpu().detach().numpy()
    np.save(os.path.join(similarity_root, 'MEN_similarity'), sim)
    return sim


def main():
    start = time.clock()
    wall_start = time.time()

    out = precompute_mental_similarities()
    np.save(os.path.join(similarity_root, 'MEN_similarity_GPU'), out)
    print('Total time: {}'.format(time.clock()-start))
    print('Total wall time: {}'.format((time.time()-wall_start)))


if __name__ == "__main__":
    main()
    a = np.load('data/similarity/MEN_similarity_GPU.npy', allow_pickle=True)
