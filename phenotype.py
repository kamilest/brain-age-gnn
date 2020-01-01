import enum


class Phenotype(enum.Enum):
    SEX = '31-0.0'  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=31
    AGE = '21003-2.0'  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21003
    FULL_TIME_EDUCATION = ['845-1.0, 845-2.0']  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=845
    FLUID_INTELLIGENCE = ['20016-0.0', '20016-1.0', '20016-2.0']  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20016
    PROSPECTIVE_MEMORY_RESULT = '20018-2.0'  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20018
