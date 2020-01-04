import numpy as np
import math

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
    return int(phenotypes.loc[subject_i, Phenotype.SEX.value[0]] == phenotypes.loc[subject_j, Phenotype.SEX.value[0]])


def custom_similarity(feature_list):
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
        for feature in feature_list:
            if np.array_equal(feature, Phenotype.MENTAL_HEALTH.value):
                # TODO compare the rest of the categories
                # First value in the mental health feature array gives the overall diagnosis as string.
                total_score += int(phenotypes.loc[subject_i, feature[0]] == phenotypes.loc[subject_j, feature[0]])
            elif len(feature) > 1:
                # handle the more/less recent values
                instance_i = feature[0]
                for f in reversed(feature):
                    if not math.isnan(phenotypes.loc[subject_i, f]):
                        instance_i = f
                        break
                instance_j = feature[0]
                for f in reversed(feature):
                    if not math.isnan(phenotypes.loc[subject_j, f]):
                        instance_j = f
                        break
                total_score += int(phenotypes.loc[subject_i, instance_i] == phenotypes.loc[subject_j, instance_j])
            else:
                total_score += int(phenotypes.loc[subject_i, feature[0]] == phenotypes.loc[subject_j, feature[0]])
        return total_score / len(feature_list)

    return get_similarity
