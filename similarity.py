import os

import numpy as np
import pandas as pd
import torch

from phenotype import Phenotype
from precompute import precompute_subject_ids

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_root = 'data'
data_phenotype = 'data/phenotype.csv'
similarity_root = 'data/similarity'

SIMILARITY_LOOKUP = 'data/similarity_lookup.pkl'
SUBJECT_IDS = 'data/subject_ids.npy'


def get_most_recent(ukb_feature, subject_id, phenotypes):
    instance = ukb_feature[0]
    for f in reversed(ukb_feature):
        if not np.isnan(phenotypes.loc[subject_id, f]):
            instance = f
            break
    return phenotypes.loc[subject_id, instance]


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

    for feature in Phenotype:
        biobank_feature = Phenotype.get_biobank_codes(feature)
        if feature == Phenotype.MENTAL_HEALTH:
            mental_to_code = Phenotype.get_mental_to_code()
            # column names for summary (total number of conditions) + 18 possible condidions: MEN0, MEN1, ..., MEN18.
            mental_feature_codes = [Phenotype.MENTAL_HEALTH.value + str(i) for i in range(19)]
            # Replace string descriptions with their codes for consistency.
            phenotype_processed.loc[:, biobank_feature[0]] = phenotype_processed[biobank_feature[0]].apply(
                lambda x: mental_to_code[x] if x in mental_to_code.keys() else None)
            # Determine if the the patient has the occurrence of a particular disease.
            si = phenotype_processed.index.to_series()
            for i in range(1, len(mental_feature_codes)):
                phenotype_processed.loc[:, Phenotype.MENTAL_HEALTH.value + str(i)] = si.apply(
                    lambda s: int(i in phenotype_processed.loc[s, biobank_feature].to_numpy().astype(bool)))
            phenotype_processed.loc[:, mental_feature_codes[0]] = si.apply(
                lambda s: int(np.sum(phenotype_processed.loc[s, mental_feature_codes[1:]])))

        elif len(biobank_feature) > 1:
            # handle the more/less recent values
            si = phenotype_processed.index.to_series().copy()
            phenotype_processed.loc[:, feature.value] = si.apply(lambda s: get_most_recent(biobank_feature, s, phenotype_processed))
        else:
            phenotype_processed.loc[:, feature.value] = phenotype_processed[biobank_feature[0]].copy()

    # Filter only the subjects used in the final dataset.
    phenotype_processed = phenotype_processed.loc[precompute_subject_ids()]

    # Return only the final feature columns (indexed by code names).
    phenotype_processed.drop(biobank_feature_list, axis=1, inplace=True)
    phenotype_processed = phenotype_processed.sort_index()

    phenotype_processed.to_pickle(SIMILARITY_LOOKUP)
    return phenotype_processed


def precompute_similarities():
    """Creates the similarity metric based on the phenotype feature list.
    If a feature has several entries in the UK Biobank, take either the most recent available estimate or, if the
    entries correspond to categories, consider the matching category values.

    The final score is an average of all the indicator scores for each feature, i.e. if two subjects have all of the
    features matching, the total score will be 1, and if none of the features match then the value will be 0. Edge
    creation then depends on the similarity threshold defined in graph construction.

    If both features are unknown, assume there is no match.
    """

    p_list = [Phenotype.AGE]
    subject_ids = np.load(SUBJECT_IDS, allow_pickle=True)
    similarity_lookup = pd.read_pickle(SIMILARITY_LOOKUP)

    for p in p_list:
        if p == Phenotype.MENTAL_HEALTH:
            mental_feature_codes = [Phenotype.MENTAL_HEALTH.value + str(i) for i in range(1, 19)]
            men = similarity_lookup.loc[subject_ids, mental_feature_codes].to_numpy()
            men = torch.tensor(men)

            sim = torch.mm(men, men.t())
            sim = sim >= 1
            sm = sim.cpu().detach().numpy()

        else:
            fea = similarity_lookup.loc[subject_ids, p.value].to_numpy()
            fea = np.expand_dims(fea, axis=0)
            fea = torch.tensor(fea)

            sim = fea.t() - fea
            sim = sim == 0
            sm = sim.cpu().detach().numpy()

        # Ignore self-similarities
        np.fill_diagonal(sm, False)
        np.save(os.path.join(similarity_root, '{}_similarity_GPU'.format(p.value)), sm)
