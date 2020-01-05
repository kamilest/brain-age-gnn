from phenotype import Phenotype


def sex_similarity(phenotypes, subject_i, subject_j):
    """
    Computes the sex similarity score, defined as an indicator whether the two subjects have the same sex.

    Args:
        phenotypes: Dataframe with phenotype values.
        subject_i: First subject.
        subject_j: Second subject.

    Returns:
        Similarity score.
    """
    return int(phenotypes.loc[subject_i, Phenotype.get_biobank_codes(Phenotype.SEX)[0]] ==
               phenotypes.loc[subject_j, Phenotype.get_biobank_codes(Phenotype.SEX)[0]])


def custom_similarity_function(feature_list):
    """
    Creates the similarity metric based on the phenotype feature list.

    If a feature has several entries in the UK Biobank, take either the most recent available estimate or, if the
    entries correspond to categories, consider the matching category values.

    The final score is an average of all the indicator scores for each feature, i.e. if two subjects have all of the
    features matching, the total score will be 1, and if none of the features match then the value will be 0. Edge
    creation then depends on the similarity threshold defined in graph construction.

    If both features are unknown, assume there is no match.
    # TODO support some deviations, e.g. if the values are in the same percentile range etc.

    Args:
        feature_list: list of features taken as Phenotype enumerated values.

    Returns:
        The similarity function taking in the phenotype list and returning the similarity score.
    """
    def get_similarity(phenotypes, subject_i, subject_j):
        total_score = 0
        if len(feature_list) == 0:
            return 0
        # TODO assert the feature set contains only Phenotype enum values.
        # TODO move this to precompute section to increase speed of pairwise comparisons.
        for feature in feature_list:
            biobank_feature = Phenotype.get_biobank_codes(feature)
            if feature == Phenotype.MENTAL_HEALTH.value:
                # TODO compare the rest of the categories
                # First value in the mental health feature array gives the overall diagnosis as string.
                total_score += int(phenotypes.loc[subject_i, biobank_feature[0]] ==
                                   phenotypes.loc[subject_j, biobank_feature[0]])
            elif len(biobank_feature) > 1:
                # handle the more/less recent values
                instance_i = biobank_feature[0]
                for f in reversed(biobank_feature):
                    if phenotypes.loc[subject_i, f] != 'NaN':
                        instance_i = f
                        break
                instance_j = biobank_feature[0]
                for f in reversed(biobank_feature):
                    if not phenotypes.loc[subject_j, f] != 'NaN':
                        instance_j = f
                        break
                total_score += int(phenotypes.loc[subject_i, instance_i] ==
                                   phenotypes.loc[subject_j, instance_j])
            else:
                total_score += int(phenotypes.loc[subject_i, biobank_feature[0]] ==
                                   phenotypes.loc[subject_j, biobank_feature[0]])
        return total_score * 1.0 / len(feature_list)

    return get_similarity
