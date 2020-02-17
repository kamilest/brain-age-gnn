import os

import numpy as np
import pandas as pd

from phenotype import Phenotype
from precompute import precompute_subject_ids

data_root = 'data'
data_phenotype = 'data/phenotype.csv'
similarity_root = 'data/similarity'

SIMILARITY_LOOKUP = 'data/similarity_lookup.pkl'
SUBJECT_IDS = 'data/subject_ids.npy'


def create_similarity_lookup():
    """Precomputes the columns of the phenotype dataset for faster subject comparison.

    :return: dataframe containing the values used for similarity comparison, row-indexed by subject ID and
    column-indexed by phenotype code name (e.g. 'AGE', 'FTE' etc.)
    """

    phenotypes = pd.read_csv(data_phenotype, sep=',')
    phenotypes.index = ['UKB' + str(eid) for eid in phenotypes['eid']]

    biobank_feature_list = []
    for feature in Phenotype:
        biobank_feature_list.extend(Phenotype.get_biobank_codes(feature))

    phenotype_processed = phenotypes[biobank_feature_list]

    def get_most_recent(ukb_feature, subject_id):
        instance = ukb_feature[0]
        for f in reversed(ukb_feature):
            if phenotypes.loc[subject_id, f] != 'NaN':
                instance = f
                break
        return phenotypes.loc[subject_id, instance]

    for feature in Phenotype:
        biobank_feature = Phenotype.get_biobank_codes(feature)
        if feature == Phenotype.MENTAL_HEALTH:
            mental_to_code = Phenotype.get_mental_to_code()
            # column names for summary + 18 possible condidions: MEN0, MEN1, ..., MEN18.
            mental_feature_codes = [Phenotype.MENTAL_HEALTH.value + str(i) for i in range(19)]
            # Replace string descriptions with their codes for consistency.
            phenotype_processed.loc[:, biobank_feature[0]] = phenotype_processed[biobank_feature[0]].apply(
                lambda x: mental_to_code[x] if x in mental_to_code.keys() else None)
            # Determine if the the patient have the occurrence of a particular disease.
            si = phenotype_processed.index.to_series()
            for i in range(1, len(mental_feature_codes)):
                phenotype_processed.loc[:, Phenotype.MENTAL_HEALTH.value + str(i)] = si.apply(
                    lambda s: int(i in phenotype_processed.loc[s, biobank_feature].to_numpy()))
            phenotype_processed.loc[:, mental_feature_codes[0]] = si.apply(
                lambda s: int(np.sum(phenotype_processed.loc[s, mental_feature_codes[1:]]) > 0))

        elif len(biobank_feature) > 1:
            # handle the more/less recent values
            si = phenotype_processed.index.to_series().copy()
            phenotype_processed.loc[:, feature.value] = si.apply(lambda s: get_most_recent(biobank_feature, s))
        else:
            phenotype_processed.loc[:, feature.value] = phenotype_processed[biobank_feature[0]].copy()

    # Filter only the subjects used in the final dataset.
    phenotype_processed = phenotype_processed.loc[precompute_subject_ids()]

    # Return only the final feature columns (indexed by code names).
    phenotype_processed.drop(biobank_feature_list, axis=1, inplace=True)
    phenotype_processed = phenotype_processed.sort_index()

    phenotype_processed.to_pickle(SIMILARITY_LOOKUP)
    return phenotype_processed


def custom_similarity_function(feature_list):
    """Creates the similarity metric based on the phenotype feature list.
    If a feature has several entries in the UK Biobank, take either the most recent available estimate or, if the
    entries correspond to categories, consider the matching category values.

    The final score is an average of all the indicator scores for each feature, i.e. if two subjects have all of the
    features matching, the total score will be 1, and if none of the features match then the value will be 0. Edge
    creation then depends on the similarity threshold defined in graph construction.

    If both features are unknown, assume there is no match.

    :param feature_list: list of features taken as Phenotype enumerated values.
    :return: similarity function taking in the phenotype list and returning the similarity score.
    """
    # TODO support some deviations, e.g. if the values are in the same percentile range etc.

    if len(feature_list) == 0:
        return lambda x: 0

    # Create look-up table of similarity features for all subjects.
    similarity_lookup = create_similarity_lookup()
    mental_feature_codes = [Phenotype.MENTAL_HEALTH.value + str(i) for i in range(19)]

    def get_similarity(subject_i, subject_j):
        total_score = 0
        for feature in feature_list:
            if feature == Phenotype.MENTAL_HEALTH:
                total_score += int(np.dot(similarity_lookup.loc[subject_i, mental_feature_codes],
                                          similarity_lookup.loc[subject_j, mental_feature_codes]) != 1)
            else:
                total_score += int(similarity_lookup.loc[subject_i, feature.value] ==
                                   similarity_lookup.loc[subject_j, feature.value])
        return total_score * 1.0 / len(feature_list)

    return get_similarity


def precompute_similarities():
    p_list = [Phenotype.AGE]
    subject_ids = np.load(SUBJECT_IDS, allow_pickle=True)
    similarity_lookup = pd.read_pickle(SIMILARITY_LOOKUP)

    for p in p_list:
        sm = np.zeros((len(subject_ids), len(subject_ids)), dtype=np.bool)

        if p == Phenotype.MENTAL_HEALTH:
            mental_feature_codes = [Phenotype.MENTAL_HEALTH.value + str(i) for i in range(19)]
            for i in range(len(subject_ids)):
                id_i = subject_ids[i]
                for j in range(i):
                    id_j = subject_ids[j]
                    sm[i, j] = sm[j, i] = int(np.dot(similarity_lookup.loc[id_i, mental_feature_codes],
                                                     similarity_lookup.loc[id_j, mental_feature_codes]) != 1)
        else:
            for i in range(len(subject_ids)):
                id_i = subject_ids[i]
                for j in range(i):
                    id_j = subject_ids[j]
                    sm[i, j] = sm[j, i] = (similarity_lookup.loc[id_i, p.value] ==
                                           similarity_lookup.loc[id_j, p.value])

        # Mask for lower triangle values.
        # mask = np.invert(np.tri(sm.shape[0], k=-1, dtype=bool))
        # m = np.ma.masked_where(mask == 1, mask)
        # lower_tri_sm = np.ma.masked_where(m, sm)
        #
        # # Flatten the similarity matrix.
        # flat_sm = lower_tri_sm.compressed()
        # assert flat_sm.size == (sm.shape[0] * (sm.shape[0] - 1)) * 0.5
        np.save(os.path.join(similarity_root, '{}_similarity'.format(p.value)), sm)
