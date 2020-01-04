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

