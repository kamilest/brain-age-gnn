import numpy as np
import sklearn
import torch


def functional_connectivities_pca(connectivities, train_idx, remaining_components=1.0, random_state=0):
    """Runs principal component analysis (PCA) on a set of flattened connectivity matrices.
    Fits to the training set and applies to the rest.

    :param connectivities: flattened functional connectivity matrices.
    :param train_idx: the boolean mask of the functional connectivity matrices belonging to the trainng set.
    :param remaining_components: fraction of reimaining components (for dimensionality reduction).
    :param random_state: random state of the PCA transform.
    :return the principal components of the functional connectivity matrices.
    """

    connectivity_pca = sklearn.decomposition.PCA(random_state=random_state)
    connectivity_pca.fit(connectivities[train_idx])
    components = connectivity_pca.transform(connectivities)
    return components[:, :int(remaining_components * components.shape[1])]


def graph_feature_transform(population_graph, pca=True, pca_remaining_components=0.01):
    """Prepares intermediate population graph with assigned training masks for training.
    Normalises and concatenates features, optionally running principal component analysis on functional MRI data.

    :param population_graph: population graph object with the assigned training validation and test masks.
    :param pca: indicates whether to run principal component analysis on functional MRI data.
    :param pca_remaining_components: the fraction of remaining components after PCA transformation (describes degree of
        dimensionality reduction.
    :return prepared population graph with updated feature tensor.
    """

    # Optional functional data preprocessing (PCA) based on the traning index.
    train_mask = population_graph.train_mask.numpy()
    if not population_graph.functional_data.empty and pca:
        functional_data = functional_connectivities_pca(population_graph.functional_data, train_mask,
                                                        remaining_components=pca_remaining_components)
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
    if not population_graph.quality_control_data.empty:
        euler_scaler = sklearn.preprocessing.StandardScaler()
        euler_scaler.fit(population_graph.quality_control_data[train_mask])
        quality_control_data = euler_scaler.transform(population_graph.quality_control_data)
    else:
        quality_control_data = population_graph.quality_control_data

    # Unify feature sets into one feature vector.
    features = np.concatenate([functional_data,
                               structural_data,
                               quality_control_data], axis=1)

    population_graph.x = torch.tensor(features, dtype=torch.float32)


def concatenate_graph_features(population_graph):
    """A utility method for concatenating all of the population graph node features.

    :param population_graph: population graph object.
    :return numpy array of concatenated population graph node features.
    """

    structural_data = []
    for structural_feature in population_graph.structural_data.keys():
        structural_data.append(population_graph.structural_data[structural_feature])

    structural_data = np.concatenate(structural_data, axis=1)

    return np.concatenate([population_graph.functional_data,
                           structural_data,
                           population_graph.quality_control_data], axis=1)


