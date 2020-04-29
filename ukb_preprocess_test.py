"""
Tests the UKB preprocessing component.
"""

import unittest

import numpy as np
import pandas as pd

from phenotype import Phenotype

SUBJECT_IDS = 'data/subject_ids.npy'
SIMILARITY_LOOKUP = 'data/similarity_lookup.pkl'
SIMILARITY_MEN = 'data/similarity/similarity_MEN.npy'

subject_ids = sorted([1192336, 1629877, 1677375, 1894259, 2875424, 2898110, 3766119, 4553519, 4581316, 4872190])
subject_ids_ukb = ['UKB{}'.format(i) for i in subject_ids]

subject_men_ids = sorted([1000028, 1000260, 1000430, 1001269, 1002352])
subject_men_ids_ukb = ['UKB{}'.format(i) for i in subject_men_ids]

sex = ['Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Female']
fte = [np.nan, 15, 17, 15, 17, 16, 16, np.nan, np.nan, np.nan]
fi = [9, 4, 9, 6, 5, 4, 8, 6, 9, 5]
labels = [63, 71, 60, 75, 54, 60, 61, 58, 61, 65]

mental = {'UKB1000028': [],
          'UKB1000260': [6],
          'UKB1000430': [15],
          'UKB1001269': [15],
          'UKB1002352': [6, 11, 15]}


class CreateSimilarityLookupTest(unittest.TestCase):
    def testSimilarityLookup(self):
        similarity_lookup = pd.read_pickle(SIMILARITY_LOOKUP)

        self.assertIsNone(np.testing.assert_array_equal(sex, similarity_lookup.loc[subject_ids_ukb, Phenotype.SEX.value]))
        self.assertIsNone(np.testing.assert_array_equal(fte, similarity_lookup.loc[subject_ids_ukb, Phenotype.FULL_TIME_EDUCATION.value]))
        self.assertIsNone(np.testing.assert_array_equal(fi, similarity_lookup.loc[subject_ids_ukb, Phenotype.FLUID_INTELLIGENCE.value]))
        self.assertIsNone(np.testing.assert_array_equal(labels, similarity_lookup.loc[subject_ids_ukb, Phenotype.AGE.value]))

    def testPrecomputeSimilarities_MEN(self):
        mental_similarity = np.load(SIMILARITY_MEN)

        full_subject_ids = np.load(SUBJECT_IDS, allow_pickle=True)
        id_mask = np.isin(full_subject_ids, subject_men_ids_ukb)
        ids = np.argwhere(id_mask).flatten()

        for i, ix in enumerate(ids):
            iter_j = iter(enumerate(ids))
            [next(iter_j) for _ in range(i+1)]
            for j, jx in iter_j:
                similarity_true = int(len(np.intersect1d(mental[subject_men_ids_ukb[i]],
                                                         mental[subject_men_ids_ukb[j]])) >= 1)
                similarity_computed = mental_similarity[ix, jx]
                self.assertEqual(similarity_true, similarity_computed)


if __name__ == '__main__':
    unittest.main()
