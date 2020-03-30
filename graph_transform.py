import numpy as np
import sklearn
import torch


def functional_connectivities_pca(connectivities, train_idx, random_state=0):
    connectivity_pca = sklearn.decomposition.PCA(random_state=random_state)
    connectivity_pca.fit(connectivities[train_idx])
    return connectivity_pca.transform(connectivities)


def graph_feature_transform(population_graph):
    # Optional functional data preprocessing (PCA) based on the traning index.
    train_mask = population_graph.train_mask.numpy()
    if not population_graph.functional_data.empty:
        functional_data = functional_connectivities_pca(population_graph.functional_data, train_mask)
    else:
        functional_data = population_graph.functional_data

    # Scaling structural data based on training index.
    # Transforming multiple structural data modalities.
    transformed_structural_features = []
    for structural_feature in population_graph.structural_data.keys():
        if not population_graph.structural_data[structural_feature].empty:
            structural_scaler = sklearn.preprocessing.StandardScaler()
            structural_scaler.fit(population_graph.structural_data[structural_feature][train_mask])
            transformed_structural_features.append(structural_scaler.transform(
                population_graph.structural_data[structural_feature]))
        else:
            transformed_structural_features.append(population_graph.structural_data[structural_feature])

    structural_data = np.concatenate(transformed_structural_features, axis=1)

    # Scaling Euler index data based on training index.
    if not population_graph.euler_data.empty:
        euler_scaler = sklearn.preprocessing.StandardScaler()
        euler_scaler.fit(population_graph.euler_data[train_mask])
        euler_data = euler_scaler.transform(population_graph.euler_data)
    else:
        euler_data = population_graph.euler_data

    # Unify feature sets into one feature vector.
    features = np.concatenate([functional_data,
                               structural_data,
                               euler_data], axis=1)

    population_graph.x = torch.tensor(features, dtype=torch.float32)


def concatenate_graph_features(population_graph):
    structural_data = []
    for structural_feature in population_graph.structural_data.keys():
        structural_data.append(population_graph.structural_data[structural_feature])

    structural_data = np.concatenate(structural_data, axis=1)

    return np.concatenate([population_graph.functional_data,
                           structural_data,
                           population_graph.euler_data], axis=1)


