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
    return int(phenotypes.loc[subject_i, Phenotype.SEX] == phenotypes.loc[subject_j, Phenotype.SEX])


def custom_similarity(feature_list):
    """
    Creates the similarity metric based on the phenotype feature list.

    If a feature has several entries in the UK Biobank, take either the most recent available estimate or, if the
    entries correspond to categories, consider the matching category values.

    The final score is an average of all the indicator scores for each feature, i.e. if two subjects have all of the
    features matching, the total score will be 1, and if none of the features match then the value will be 0. Edge
    creation then depends on the similarity threshold defined in graph construction.
    # TODO support some error, e.g. if the values are in the same percentile range etc.

    Args:
        feature_list: list of features taken as Phenotype enumerated values.

    Returns:
        The similarity function taking in the phenotype list and returning the similarity score.
    """

    total_score = 0
    for feature in feature_list:
        if feature.value == Phenotype.MENTAL_HEALTH:
            # handle categories
            pass
        elif isinstance(feature.value, list):
            # handle the more/less recent values
            pass
        else:
            # compare normally
            pass

    return total_score
