import pandas as pd

data_root = 'data'
data_timeseries = 'data/raw_ts'
data_phenotype = 'data/phenotype.csv'
data_ct = 'data/CT.csv'
data_sa = 'data/SA.csv'
data_vol = 'data/Vol.csv'
data_euler = 'data/Euler.csv'
data_computed_fcms = 'data/processed_ts'

# Exclude the following raw timeseries due to incorrect size.
EXCLUDED_UKB_IDS = ['UKB2203847', 'UKB2208238', 'UKB2697888']


# extract_phenotypes(['31-0.0', '21003-2.0'], ['UKB1000028', 'UKB1000133'])
def extract_phenotypes(subject_ids, uid_list=None):
    if uid_list is None:
        uid_list = ['eid']
    else:
        uid_list.append('eid')
    phenotype = pd.read_csv(data_phenotype, sep=',')
    subject_ids_no_UKB = [int(i[3:]) for i in subject_ids]

    # Extract data for relevant subject IDs.
    subject_phenotype = phenotype[phenotype['eid'].isin(subject_ids_no_UKB)].copy()

    if len(subject_phenotype) != len(subject_ids):
        print('{} entries had phenotypic data missing.'.format(len(subject_ids) - len(subject_phenotype)))

    # Extract relevant UIDs.
    if len(uid_list) > 1:
        subject_phenotype = subject_phenotype[uid_list]

    # Add UKB prefix back to the index.
    subject_phenotype.index = ['UKB' + str(eid) for eid in subject_phenotype['eid']]
    subject_phenotype.sort_index()

    return subject_phenotype


def extract_structural(subject_ids, type):
    if type == 'cortical_thickness':
        data = data_ct
    elif type == 'surface_area':
        data = data_sa
    elif type == 'volume':
        data = data_vol
    else:
        return pd.DataFrame(pd.np.empty((len(subject_ids), 0)))

    ct = pd.read_csv(data, sep=',', quotechar='\"')

    # Extract data for relevant subject IDs.
    subject_ct = ct[ct['NewID'].isin(subject_ids)].copy()

    assert(len(subject_ids) - len(subject_ct) == 0)
    if len(subject_ct) != len(subject_ids):
        print('{} entries had {} data missing.'.format(len(subject_ids) - len(subject_ct), type))

    subject_ct = subject_ct.drop(subject_ct.columns[0], axis=1)
    subject_ct = subject_ct.drop(['lh_???', 'rh_???'], axis=1)

    subject_ct.index = subject_ct['NewID']
    subject_ct = subject_ct.drop(['NewID'], axis=1)
    subject_ct = subject_ct.sort_index()

    return subject_ct


def extract_euler(subject_ids):
    euler = pd.read_csv(data_euler, sep=',', quotechar='\"')

    # Extract data for relevant subject IDs.
    subject_euler = euler[euler['eid'].isin(subject_ids)].copy()
    assert (len(subject_ids) - len(subject_euler) == 0)

    subject_euler.index = subject_euler['eid']
    subject_euler = subject_euler.drop(['eid', 'oldID'], axis=1)
    subject_euler = subject_euler.sort_index()

    return subject_euler
