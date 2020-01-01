# Graph construction phenotypic parameters.
# http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=31
SEX_UID = '31-0.0'


def sex_similarity(phenotypes, subject_i, subject_j):
    """
    Computes the similarity score between two subjects.

    Args:
        phenotypes: Dataframe with phenotype values.
        subject_i: First subject.
        subject_j: Second subject.

    Returns:
        Similarity score.
    """
    return int(phenotypes.loc[subject_i, SEX_UID] == phenotypes.loc[subject_j, SEX_UID])

