import argparse
import ast
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import brain_gcn
import gnn_train_evaluate
import graph_construct
import graph_transform
from phenotype import Phenotype

graph_root = 'data/graph'
GRAPH_NAMES = sorted(os.listdir(graph_root))

logdir = './runs/{}'.format(datetime.now().strftime('%Y-%m-%d'))
Path(logdir).mkdir(parents=True, exist_ok=True)

# ARGUMENT PARSING
parser = argparse.ArgumentParser(description='Brain age graph neural network.')

# Neural network parameters
parser.add_argument('--model', default='gcn', type=str, help='Type of model (options: gcn, fc)')
parser.add_argument('--epochs', default=5000, type=int, help='Number of epochs (default 5000)')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='Learning rate (default 5e-4)')
parser.add_argument('--dropout', default=0, type=float, help='Dropout (default 0)')
parser.add_argument('--weight_decay', default=0, type=float, help='Weight decay (default 0)')
parser.add_argument('--n_conv_layers', default=1, type=int, help='Number of graph convolutional layers (default: 1)')
parser.add_argument('--layer_sizes', default='[364, 364, 512, 256, 256, 1]', type=str, help='Sizes of layers')

# Population graph parameters
parser.add_argument('--functional', default=0, type=bool)
parser.add_argument('--structural', default=1, type=bool)
parser.add_argument('--euler', default=1, type=bool)
parser.add_argument('--similarity', default="(['SEX', 'ICD10', 'FTE', 'NEU'], 0.8)", type=str)

args = parser.parse_args()

functional = args.functional
structural = args.structural
euler = args.euler
similarity_feature_set = [Phenotype(i) for i in ast.literal_eval(args.similarity)[0]]
similarity_threshold = ast.literal_eval(args.similarity)[1]

graph_name = graph_construct.get_graph_name(functional=functional,
                                            structural=structural,
                                            euler=euler,
                                            similarity_feature_set=similarity_feature_set,
                                            similarity_threshold=similarity_threshold)

if graph_name not in GRAPH_NAMES:
    graph_construct.construct_population_graph(similarity_feature_set=similarity_feature_set,
                                               similarity_threshold=similarity_threshold,
                                               functional=functional,
                                               structural=structural,
                                               euler=euler)

if args.model == 'gcn' or args.model == 'gat':
    n_conv_layers = args.n_conv_layers
else:
    n_conv_layers = 0


torch.manual_seed(99)
np.random.seed(0)

population_graph = graph_construct.load_population_graph(graph_root, graph_name)
fold = gnn_train_evaluate.get_stratified_subject_split(population_graph)
gnn_train_evaluate.set_training_masks(population_graph, *fold)
graph_transform.graph_feature_transform(population_graph)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

brain_gcn.gcn_train_with_cross_validation(population_graph, device,
                                          n_conv_layers=n_conv_layers,
                                          layer_sizes=ast.literal_eval(args.layer_sizes),
                                          lr=args.learning_rate,
                                          weight_decay=args.weight_decay,
                                          dropout_p=args.dropout,
                                          epochs=args.epochs,
                                          n_folds=5,
                                          patience=100)
