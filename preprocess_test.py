import unittest

import numpy as np

import precompute
import preprocess
from phenotype import Phenotype

subject_ids = [1192336, 1629877, 1677375, 1894259, 2875424, 2898110, 3766119, 4553519, 4581316, 4872190]
subject_ids_ukb = ['UKB{}'.format(i) for i in subject_ids]

sex = ['M', 'M', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'F']
fte = [np.nan, 15, 17, 15, 17, 16, 16, np.nan, np.nan, np.nan]
fi = [9, 4, 9, 6, 5, 4, 8, 6, 9, 5]
labels = [63, 71, 60, 75, 54, 60, 61, 58, 61, 65]

# Structural data
ct = precompute.extract_structural(subject_ids_ukb, 'cortical_thickness')
sa = precompute.extract_structural(subject_ids_ukb, 'surface_area')
gmv = precompute.extract_structural(subject_ids_ukb, 'volume')
euler = precompute.extract_euler(subject_ids_ukb)

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


class ConstructPopulationGraphTest(unittest.TestCase):
    def testConstructPopulationGraph_ToyGraph(self):
        graph = preprocess.construct_population_graph([Phenotype.SEX],
                                                      subject_ids=subject_ids_ukb,
                                                      age_filtering=False,
                                                      save=False)
        actual_labels = graph.y.numpy().flatten()
        self.assertTrue(np.array_equal(actual_labels, labels))
        self.assertTrue(np.array_equal(graph.subject_index, subject_ids_ukb))

        # Comparing structural data
        actual_ct = graph.structural_data['cortical_thickness']
        actual_sa = graph.structural_data['surface_area']
        actual_gmv = graph.structural_data['volume']
        actual_euler = graph.euler_data

        self.assertTrue(np.array_equal(actual_ct, ct))
        self.assertTrue(np.array_equal(actual_sa, sa))
        self.assertTrue(np.array_equal(actual_gmv, gmv))
        self.assertTrue(np.array_equal(actual_euler, euler))


if __name__ == '__main__':
    unittest.main()
