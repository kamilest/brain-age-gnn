import enum

# http://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=1401
MENTAL_TO_CODE = {'Anxiety, nerves or GAD': 15,
                  'Panic attacks': 6,
                  'Social Anxiety or phobia': 1,
                  'Other phobia': 5,
                  'OCD': 7,
                  'Other type of Psychosis or psychotic ilness': 3,
                  'Mania, hypomania, bipolar or manic-depression': 10,
                  'Anorexia nervosa': 16,
                  'Prefer not to answer(Group B)': -819,
                  'Other personality disorder': 4,
                  'Prefer not to answer(Group A)': -818,
                  'Bulimia nervosa': 12,
                  'Schizophrenia': 2,
                  'Autism, Asperger or binge-eating': 14,
                  'Agoraphobia': 17,
                  'Psychological over-eating or binge-eating': 13,
                  'ADD/ADHD': 18}


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
    ICD10 = 'ICD10'

    @staticmethod
    def get_biobank_codes(feature):
        biobank_features = {
            Phenotype.SEX: ['31-0.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=31
            Phenotype.AGE: ['21003-2.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21003
            Phenotype.FULL_TIME_EDUCATION: ['845-0.0', '845-1.0', '845-2.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=845
            Phenotype.FLUID_INTELLIGENCE: ['20016-0.0', '20016-1.0', '20016-2.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20016
            Phenotype.PROSPECTIVE_MEMORY_RESULT: ['20018-2.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20018
            Phenotype.MENTAL_HEALTH: ['20544-0.' + str(i) for i in range(1, 17)],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20544
            Phenotype.BIPOLAR_DISORDER_STATUS: ['20122-0.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20122
            Phenotype.NEUROTICISM_SCORE: ['20127-0.0'],  # http://biobank.ndph.ox.ac.,uk/showcase/field.cgi?id=20127
            Phenotype.SMOKING_STATUS: ['20116-2.0'],  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=20116
            Phenotype.ICD10: ['X41270.0.' + str(i) for i in range(0, 213)]  # http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=41270
        }

        return biobank_features[feature]

    @staticmethod
    def get_mental_to_code():
        return MENTAL_TO_CODE

