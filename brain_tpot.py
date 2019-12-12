import numpy as np
import os

import sklearn
from tpot import TPOTRegressor

import preprocess
import precompute

# http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=31
SEX_UID = '31-0.0'
# http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21003
AGE_UID = '21003-2.0'

subject_ids = preprocess.get_subject_ids(1000)

phenotypes = precompute.extract_phenotypes([SEX_UID, AGE_UID], subject_ids)
connectivities = np.array([preprocess.get_functional_connectivity(i) for i in phenotypes.index])

labels = np.array(phenotypes[AGE_UID].tolist())


num_train = int(len(phenotypes) * 0.85)
num_validate = int(len(phenotypes) * 0.05)

train_val_idx = np.random.choice(range(len(phenotypes)), num_train + num_validate, replace=False)
train_idx = np.random.choice(train_val_idx, num_train, replace=False)
validate_idx = list(set(train_val_idx) - set(train_idx))
test_idx = list(set(range(len(phenotypes))) - set(train_val_idx))

X_train, X_test = connectivities[train_idx], connectivities[validate_idx]
y_train, y_test = labels[train_idx], labels[validate_idx]


pipeline_optimizer = TPOTRegressor(generations=10, n_jobs=-1, max_time_mins=120, verbosity=3)

pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('tpot_exported_pipeline.py')


