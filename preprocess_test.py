import unittest

import numpy as np

import preprocess
from phenotype import Phenotype

subject_ids = [4872190, 4581316, 1677375, 3766119, 2875424, 1894259, 2898110, 4553519, 1192336, 1629877]
subject_ids_ukb = ['UKB{}'.format(i) for i in subject_ids]

sex = ['M', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'F']
fte = [np.nan, 15, 17, 15, 17, 16, 16, np.nan, np.nan, np.nan]
fi = [9, 4, 9, 6, 5, 4, 8, 6, 9, 5]


class ConstructEdgeListTest(unittest.TestCase):
    def testConstructEdgeList_SexSimilarity(self):
        edge_list = preprocess.construct_edge_list(subject_ids_ukb, [Phenotype.SEX])
        true_edge_list = []
        for i in range(len(subject_ids)):
            for j in range(len(subject_ids)):
                if i != j and sex[i] == sex[j]:
                    true_edge_list.append([i, j])

        self.assertTrue(np.array_equal(edge_list, true_edge_list))

    def testConstructEdgeList_FullTimeEducationSimilarity(self):
        edge_list = preprocess.construct_edge_list(subject_ids_ukb, [Phenotype.FULL_TIME_EDUCATION])
        true_edge_list = []
        for i in range(len(subject_ids)):
            for j in range(len(subject_ids)):
                if i != j and fte[i] == fte[j]:
                    true_edge_list.append([i, j])

        self.assertTrue(np.array_equal(edge_list, true_edge_list))

    def testConstructEdgeList_FluidIntelligenceSimilarity(self):
        edge_list = preprocess.construct_edge_list(subject_ids_ukb, [Phenotype.FLUID_INTELLIGENCE])
        true_edge_list = []
        for i in range(len(subject_ids)):
            for j in range(len(subject_ids)):
                if i != j and fi[i] == fi[j]:
                    true_edge_list.append([i, j])

        self.assertTrue(np.array_equal(edge_list, true_edge_list))

    def testConstructEdgeList_SexAndFluidIntelligenceSimilarity(self):
        edge_list = preprocess.construct_edge_list(subject_ids_ukb,
                                                   [Phenotype.SEX, Phenotype.FLUID_INTELLIGENCE],
                                                   similarity_threshold=1)
        true_edge_list = []
        for i in range(len(subject_ids)):
            for j in range(len(subject_ids)):
                if i != j and fi[i] == fi[j] and sex[i] == sex[j]:
                    true_edge_list.append([i, j])

        self.assertTrue(np.array_equal(edge_list, true_edge_list))

    def testConstructEdgeList_SexOrFluidIntelligenceSimilarity(self):
        edge_list = preprocess.construct_edge_list(subject_ids_ukb,
                                                   [Phenotype.SEX, Phenotype.FLUID_INTELLIGENCE],
                                                   similarity_threshold=0.5)
        true_edge_list = []
        for i in range(len(subject_ids)):
            for j in range(len(subject_ids)):
                if i != j and fi[i] == fi[j] or sex[i] == sex[j]:
                    true_edge_list.append([i, j])

        self.assertTrue(np.array_equal(edge_list, true_edge_list))


if __name__ == '__main__':
    unittest.main()
