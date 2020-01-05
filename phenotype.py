import enum


class Phenotype(enum.Enum):
    SEX = 'SEX'
    AGE = 'AGE'
    FULL_TIME_EDUCATION = 'FTE'
    FLUID_INTELLIGENCE = 'FI'
    PROSPECTIVE_MEMORY_RESULT = 'MEM'
    MENTAL_HEALTH = 'MEN'
    BIPOLAR_DISORDER_STATUS = 'BIP'
    NEUROTICISM_SCORE = 'NEU'
    SMOKING_STATUS = 'SMO'


biobank_features = {
    Phenotype.SEX: ['31-0.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=31
    Phenotype.AGE: ['21003-2.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21003
    Phenotype.FULL_TIME_EDUCATION: ['845-0.0', '845-1.0', '845-2.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=845
    Phenotype.FLUID_INTELLIGENCE: ['20016-0.0', '20016-1.0', '20016-2.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20016
    Phenotype.PROSPECTIVE_MEMORY_RESULT: ['20018-2.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20018
    Phenotype.MENTAL_HEALTH: ['20544-0.' + str(i) for i in range(1, 17)],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20544
    Phenotype.BIPOLAR_DISORDER_STATUS: ['20122-0.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20122
    Phenotype.NEUROTICISM_SCORE: ['20127-0.0'],  # http://biobank.ndph.ox.ac.,uk/showcase/field.cgi?id=20127
    Phenotype.SMOKING_STATUS: ['20116-2.0']  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20116
}
