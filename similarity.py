import pandas as pd

from phenotype import Phenotype

data_phenotype = 'data/phenotype.csv'


def get_similarity_lookup(feature_list):
    """Precomputes the columns of the phenotype dataset for faster subject comparison.

    :return: dataframe containing the values used for similarity comparison, row-indexed by subject ID and
    column-indexed by phenotype code name (e.g. 'AGE', 'FTE' etc.)
    """
    # TODO without referring to the phenotype in get_similarity, just the subjects.

    phenotypes = pd.read_csv(data_phenotype, sep=',')
    phenotypes.index = ['UKB' + str(eid) for eid in phenotypes['eid']]

    biobank_feature_list = []
    for feature in feature_list:
        biobank_feature_list.extend(Phenotype.get_biobank_codes(feature))

    phenotype_processed = phenotypes[biobank_feature_list]

    def get_most_recent(ukb_feature, subject_id):
        instance = ukb_feature[0]
        for f in reversed(ukb_feature):
            if phenotypes.loc[subject_id, f] != 'NaN':
                instance = f
                break
        return phenotypes.loc[subject_id, instance]

    for feature in feature_list:
        if feature in Phenotype:
            biobank_feature = Phenotype.get_biobank_codes(feature)
            if feature == Phenotype.MENTAL_HEALTH:
                # TODO compare the rest of the categories
                # First value in the mental health feature array gives the overall diagnosis as string.
                phenotype_processed.loc[:, feature.value] = phenotype_processed[biobank_feature[0]].copy()
            elif len(biobank_feature) > 1:
                # handle the more/less recent values
                si = phenotype_processed.index.to_series().copy()
                phenotype_processed.loc[:, feature.value] = si.apply(lambda s: get_most_recent(biobank_feature, s))
            else:
                phenotype_processed.loc[:, feature.value] = phenotype_processed[biobank_feature[0]].copy()

    # Return only the final feature columns (indexed by code names).
    phenotype_processed.drop(biobank_feature_list, axis=1, inplace=True)
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
    similarity_lookup = get_similarity_lookup(feature_list)

    def get_similarity(subject_i, subject_j):
        total_score = 0
        for feature in feature_list:
            total_score += int(similarity_lookup.loc[subject_i, feature.value] ==
                               similarity_lookup.loc[subject_j, feature.value])
        return total_score * 1.0 / len(feature_list)

    return get_similarity
