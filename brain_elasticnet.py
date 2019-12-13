import numpy as np
import os

import sklearn

import preprocess
import precompute

# http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=31
SEX_UID = '31-0.0'
# http://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=21003
AGE_UID = '21003-2.0'

subject_ids = preprocess.get_subject_ids(1000)

phenotypes = precompute.extract_phenotypes([SEX_UID, AGE_UID], subject_ids)
connectivities = precompute.extract_cortical_thickness(phenotypes.index).to_numpy()

labels = np.array(phenotypes[AGE_UID].tolist())


num_train = int(len(phenotypes) * 0.85)
num_validate = int(len(phenotypes) * 0.05)

train_val_idx = np.random.choice(range(len(phenotypes)), num_train + num_validate, replace=False)
train_idx = np.random.choice(train_val_idx, num_train, replace=False)
validate_idx = list(set(train_val_idx) - set(train_idx))
test_idx = list(set(range(len(phenotypes))) - set(train_val_idx))

X_train, X_test = connectivities[train_idx], connectivities[validate_idx]
y_train, y_test = labels[train_idx], labels[validate_idx]

# connectivity_pca = sklearn.decomposition.PCA(random_state=42)
# connectivity_pca.fit(connectivities[train_idx])
# connectivities_transformed = connectivity_pca.transform(connectivities)

scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(connectivities[train_idx])
connectivities_transformed = scaler.transform(connectivities)

X_train_prepared, X_test_prepared = connectivities_transformed[train_idx], connectivities_transformed[validate_idx]

elastic_net = sklearn.linear_model.ElasticNet(l1_ratio=0.75, random_state=42)
elastic_net.fit(X_train_prepared, y_train)
print(elastic_net.score(X_test_prepared, y_test))

