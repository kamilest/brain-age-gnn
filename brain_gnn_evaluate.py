"""
Graph neural network evaluation component.

Provides functions for altering the graph by addning Gaussian noise to nodes, permuting the node features, removing
    edges at random as part of permutation and robustness tests.
"""

import ast
import gc
import os

import numpy as np
import torch
import wandb
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import brain_gnn_train
import graph_construct
import graph_transform
from brain_gnn import ConvTypes, BrainGCN, BrainGAT
from phenotype import Phenotype

graph_root = 'data/graph'
model_root = 'data/model'

GRAPH_NAMES = sorted(os.listdir(graph_root))

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def add_population_graph_noise(population_graph, p, noise_amplitude=0.5, random_state=0):
    """Adds white Gaussian noise to the nodes of the population graph, modifying the feature vector.

    :param population_graph: population graph.
    :param p: proportion of training nodes with added noise.
    :param noise_amplitude: the variance of white noise.
    :param random_state: random state determining which nodes will get added noise.
    """

    nodes = population_graph.x.numpy().copy()
    train_idx = np.where(population_graph.train_mask.numpy())[0]

    np.random.seed(random_state)
    noisy_train_idx = np.random.choice(train_idx, round(len(train_idx) * p), replace=False)

    for i in noisy_train_idx:
        nodes[i] += np.random.normal(0, noise_amplitude, len(nodes[i]))

    scaler = StandardScaler()
    scaler.fit(nodes[population_graph.train_mask])
    nodes = scaler.transform(nodes)

    population_graph.x = torch.tensor(nodes)


def permute_population_graph_features(population_graph, p, random_state=0, train_set_only=True):
    """Adds white Gaussian noise to the nodes of the population graph, modifying the feature vector.

    :param population_graph: population graph.
    :param p: proportion of training nodes with permuted features.
    :param random_state: random state determining which nodes will get added noise.
    :param train_set_only: whether to permute features only in nodes belonging to the training set.
    """

    nodes = population_graph.x.numpy().copy()
    train_idx = np.where(population_graph.train_mask.numpy())[0]

    np.random.seed(random_state)
    noisy_train_idx = np.random.choice(train_idx, round(len(train_idx) * p), replace=False)

    for i in noisy_train_idx:
        nodes[i] = np.random.permutation(nodes[i])

    population_graph.x = torch.tensor(nodes)


def remove_population_graph_edges(population_graph, p, random_state=0):
    """Removes graph edges with probability p.

    :param population_graph: path to the population graph file.
    :param p: proportion of the edges removed.
    :param random_state: the seed determining which edges are removed.
    """
    if hasattr(population_graph, 'original_edge_index'):
        edges = np.transpose(population_graph.original_edge_index.numpy().copy())
    else:
        edges = np.transpose(population_graph.edge_index.numpy().copy())
    unique_edges = np.unique([list(x) for x in (frozenset(y) for y in edges)], axis=0)

    np.random.seed(random_state)
    idx = np.random.choice(range(len(unique_edges)), replace=False, size=round(len(unique_edges) * p))
    unique_edges = np.delete(unique_edges, idx, axis=0)

    v_list, w_list = [], []
    for edge in unique_edges:
        v_list.extend([edge[0], edge[1]])
        w_list.extend([edge[1], edge[0]])

    if not hasattr(population_graph, 'original_edge_index'):
        population_graph.original_edge_index = population_graph.edge_index.clone()
    population_graph.edge_index = torch.tensor([v_list, w_list], dtype=torch.long)


def permute_population_graph_labels(population_graph, random_state=0):
    """Shuffles the labels of the population graph.

    :param population_graph: path to the population graph file.
    :param random_state: the seed determining how labels are shuffled.
    """
    if hasattr(population_graph, 'original_y'):
        y = population_graph.original_y.numpy().copy()
    else:
        y = population_graph.y.numpy().copy()

    np.random.seed(random_state)
    permuted_y = np.random.permutation(y)

    if not hasattr(population_graph, 'original_y'):
        population_graph.original_y = population_graph.y.clone()
    population_graph.y = torch.tensor(permuted_y, dtype=torch.long)


def evaluate_test_set_performance(model_dir):
    """Measures the test set performance of the model under the specified model directory.

    :param model_dir: directory containing the model state dictionaries for each fold and the model
        configuration (including the population graph parameterisation)
    :return: the test set performance for each fold.
    """

    with open(os.path.join(model_dir, 'config.yaml')) as file:
        cfg = yaml.full_load(file)

        graph_name = cfg['graph_name']['value']
        conv_type = cfg['model']['value']

        n_conv_layers = cfg['n_conv_layers']['value']
        layer_sizes = ast.literal_eval(cfg['layer_sizes']['value'])
        dropout_p = cfg['dropout']['value']

        similarity_feature_set = [Phenotype(i) for i in ast.literal_eval(cfg['similarity']['value'])[0]]
        similarity_threshold = ast.literal_eval(cfg['similarity']['value'])[1]

    if graph_name not in GRAPH_NAMES:
        graph_construct.construct_population_graph(similarity_feature_set=similarity_feature_set,
                                                   similarity_threshold=similarity_threshold,
                                                   functional=False,
                                                   structural=True,
                                                   euler=True)

    graph = graph_construct.load_population_graph(graph_root, graph_name)

    folds = brain_gnn_train.get_cv_subject_split(graph, n_folds=5)
    results = {}

    for i, fold in enumerate(folds):
        brain_gnn_train.set_training_masks(graph, *fold)
        graph_transform.graph_feature_transform(graph)

        if ConvTypes(conv_type) == ConvTypes.GCN:
            model = BrainGCN(graph.num_node_features, n_conv_layers, layer_sizes, dropout_p)
        else:
            model = BrainGAT(graph.num_node_features, n_conv_layers, layer_sizes, dropout_p)

        model.load_state_dict(torch.load(os.path.join(model_dir, 'fold-{}_state_dict.pt'.format(i))))
        model = model.to(device)
        model.eval()

        data = graph.to(device)
        model = model(data)

        predicted = model[data.test_mask].cpu()
        actual = graph.y[data.test_mask].cpu()

        r2 = r2_score(actual.detach().numpy(), predicted.detach().numpy())
        r = pearsonr(actual.detach().numpy().flatten(), predicted.detach().numpy().flatten())
        results['fold_{}'.format(i)] = {'r': [x.item() for x in r], 'r2': r2.item()}
        mse = mean_squared_error(actual.detach().numpy(), predicted.detach().numpy())
        results=mse
        break

    return results


def evaluate_noise_performance(model_dir, noise_type='node'):
    """Measures the test set performance of the model under the specified model directory when noise is added.

    :param model_dir: directory containing the model state dictionaries for each fold and the model
        configuration (including the population graph parameterisation)
    :param noise_type: 'node', 'node_feature_permutation' or 'edge'.
    :return: the dictionary of results under five different random seeds and increasing probabilities of added noise.
    """

    with open(os.path.join(model_dir, 'config.yaml')) as file:
        cfg = yaml.full_load(file)

        graph_name = cfg['graph_name']['value']
        conv_type = cfg['model']['value']

        n_conv_layers = cfg['n_conv_layers']['value']
        layer_sizes = ast.literal_eval(cfg['layer_sizes']['value'])
        dropout_p = cfg['dropout']['value']

        lr = cfg['learning_rate']['value']
        weight_decay = cfg['weight_decay']['value']

        similarity_feature_set = [Phenotype(i) for i in ast.literal_eval(cfg['similarity']['value'])[0]]
        similarity_threshold = ast.literal_eval(cfg['similarity']['value'])[1]

    if graph_name not in GRAPH_NAMES:
        graph_construct.construct_population_graph(similarity_feature_set=similarity_feature_set,
                                                   similarity_threshold=similarity_threshold,
                                                   functional=False,
                                                   structural=True,
                                                   euler=True)

    graph = graph_construct.load_population_graph(graph_root, graph_name)

    folds = brain_gnn_train.get_cv_subject_split(graph, n_folds=5)
    fold = folds[0]
    results = {}

    for i in range(1, 5):
        brain_gnn_train.set_training_masks(graph, *fold)
        results_fold = {}

        for p in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 0.95]:
            graph.to('cpu')
            graph_transform.graph_feature_transform(graph)
            if noise_type == 'node':
                add_population_graph_noise(graph, p, random_state=i)
            if noise_type == 'edge':
                remove_population_graph_edges(graph, p, random_state=i)
            if noise_type == 'node-feature-permutation':
                permute_population_graph_features(graph, p, random_state=i)


            data = graph.to(device)
            epochs = 10000
            model, _ = brain_gnn_train.train(conv_type, graph, device, n_conv_layers, layer_sizes, epochs, lr,
                                             dropout_p, weight_decay, patience=100)
            model.eval()
            model = model(data)

            predicted = model[data.test_mask].cpu()
            actual = data.y[data.test_mask].cpu()
            r2 = r2_score(actual.detach().numpy(), predicted.detach().numpy())
            r = pearsonr(actual.detach().numpy().flatten(), predicted.detach().numpy().flatten())
            results_fold['p={}_metric=r'.format(p)] = [x.item() for x in r][0]
            wandb.run.summary['{}_{}_{}_p={}_metric=r'.format(conv_type, noise_type, i, p)] = [x.item() for x in r][0]
            results_fold['p={}_metric=r2'.format(p)] = r2.item()
            wandb.run.summary['{}_{}_{}_p={}_metric=r2'.format(conv_type, noise_type, i, p)] = r2.item()

            gc.collect()

        results['{}_{}_{}'.format(conv_type, noise_type, i)] = results_fold

    return results


def label_permutation_test(model_dir):
    """Permutation test measuring the performance of the model when the labels are shuffled.

    :param model_dir: directory containing the model state dictionaries for each fold and the model
        configuration (including the population graph parameterisation)
    :return: the test set performance for each permutation.
    """

    with open(os.path.join(model_dir, 'config.yaml')) as file:
        cfg = yaml.full_load(file)

        graph_name = cfg['graph_name']['value']
        conv_type = cfg['model']['value']

        n_conv_layers = cfg['n_conv_layers']['value']
        layer_sizes = ast.literal_eval(cfg['layer_sizes']['value'])
        dropout_p = cfg['dropout']['value']

        similarity_feature_set = [Phenotype(i) for i in ast.literal_eval(cfg['similarity']['value'])[0]]
        similarity_threshold = ast.literal_eval(cfg['similarity']['value'])[1]

    if graph_name not in GRAPH_NAMES:
        graph_construct.construct_population_graph(similarity_feature_set=similarity_feature_set,
                                                   similarity_threshold=similarity_threshold,
                                                   functional=False,
                                                   structural=True,
                                                   euler=True)

    graph = graph_construct.load_population_graph(graph_root, graph_name)

    folds = brain_gnn_train.get_cv_subject_split(graph, n_folds=5)
    fold = folds[0]
    brain_gnn_train.set_training_masks(graph, *fold)
    graph_transform.graph_feature_transform(graph)

    rs = []
    r2s = []
    mses = []

    for i in range(1000):
        graph.to('cpu')
        permute_population_graph_labels(graph, i)

        if ConvTypes(conv_type) == ConvTypes.GCN:
            model = BrainGCN(graph.num_node_features, n_conv_layers, layer_sizes, dropout_p)
        else:
            model = BrainGAT(graph.num_node_features, n_conv_layers, layer_sizes, dropout_p)

        model.load_state_dict(torch.load(os.path.join(model_dir, 'fold-{}_state_dict.pt'.format(0))))
        model = model.to(device)

        data = graph.to(device)
        model.eval()
        model = model(data)

        predicted = model[data.test_mask].cpu()
        actual = graph.y[data.test_mask].cpu()

        r2 = r2_score(actual.detach().numpy(), predicted.detach().numpy())
        r = pearsonr(actual.detach().numpy().flatten(), predicted.detach().numpy().flatten())
        mse = mean_squared_error(actual.detach().numpy(), predicted.detach().numpy())

        rs.append(r[0])
        r2s.append(r2)
        mses.append(mse)
        print(r[0], r2, mse)

    np.save(os.path.join('notebooks', 'permutations_{}_{}'.format(conv_type, 'r')), rs)
    np.save(os.path.join('notebooks', 'permutations_{}_{}'.format(conv_type, 'r2')), r2s)
    np.save(os.path.join('notebooks', 'permutations_{}_{}'.format(conv_type, 'mse')), mses)

    return [rs, r2s]


wandb.init(project="brain-age-gnn", reinit=True)
wandb.save("*.pt")
results_gcn = evaluate_noise_performance(os.path.join(model_root, 'gat'), 'node-feature-permutation')
with open(os.path.join(model_root, 'gat', 'results_node-feature-permutation.yaml'), 'w+') as file:
    yaml.dump(results_gcn, file)

# results = label_permutation_test(os.path.join(model_root, 'gat'))
# with open(os.path.join(model_root, 'gat', 'results_label_permutation.yaml'), 'w+') as file:
#     yaml.dump(results, file)