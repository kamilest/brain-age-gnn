import unittest

import numpy as np
import pandas as pd

from phenotype import Phenotype

SIMILARITY_LOOKUP = 'data/similarity_lookup.pkl'

subject_ids = sorted([1192336, 1629877, 1677375, 1894259, 2875424, 2898110, 3766119, 4553519, 4581316, 4872190])
subject_ids_ukb = ['UKB{}'.format(i) for i in subject_ids]

sex = ['Male', 'Male', 'Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Female', 'Female']
fte = [np.nan, 15, 17, 15, 17, 16, 16, np.nan, np.nan, np.nan]
fi = [9, 4, 9, 6, 5, 4, 8, 6, 9, 5]
labels = [63, 71, 60, 75, 54, 60, 61, 58, 61, 65]


class CreateSimilarityLookupTest(unittest.TestCase):
    def testSimilarityLookup(self):
        similarity_lookup = pd.read_pickle(SIMILARITY_LOOKUP)

        self.assertIsNone(np.testing.assert_array_equal(sex, similarity_lookup.loc[subject_ids_ukb, Phenotype.SEX.value]))
        self.assertIsNone(np.testing.assert_array_equal(fte, similarity_lookup.loc[subject_ids_ukb, Phenotype.FULL_TIME_EDUCATION.value]))
        self.assertIsNone(np.testing.assert_array_equal(fi, similarity_lookup.loc[subject_ids_ukb, Phenotype.FLUID_INTELLIGENCE.value]))
        self.assertIsNone(np.testing.assert_array_equal(labels, similarity_lookup.loc[subject_ids_ukb, Phenotype.AGE]))


if __name__ == '__main__':
    unittest.main()
