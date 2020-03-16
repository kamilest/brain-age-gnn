import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb

import brain_gcn
import evaluate
import preprocess

graph_root = 'data/graph'
GRAPH_NAMES = sorted(os.listdir(graph_root))

logdir = './runs/{}'.format(datetime.now().strftime('%Y-%m-%d'))
Path(logdir).mkdir(parents=True, exist_ok=True)

# ARGUMENT PARSING

parser = argparse.ArgumentParser(description='Brain age graph neural network.')
# parser.add_argument('--graph_index', default=0, type=int, help='Graph construction (as index to predefined name array)')
parser.add_argument('--graph_name', default=GRAPH_NAMES[0], type=str, help='Graph structure')
parser.add_argument('--model', default='gcn', type=str, help='Type of model (options: gcn, fc)')
parser.add_argument('--epochs', default=5000, type=int, help='Number of epochs (default 5000)')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='Learning rate (default 5e-4)')
parser.add_argument('--weight_decay', default=0, type=float, help='Weight decay (default 0)')

parser.add_argument('--n_conv_layers', default=1, type=int, help='Number of graph convolutional layers (default: 1)')
parser.add_argument('--layer_sizes', default='[364, 364, 512, 256, 1]', type=str, help='Sizes of layers')

args = parser.parse_args()
# graph_index = args.graph_index
# graph_name = GRAPH_NAMES[graph_index]
graph_name = args.graph_name

if args.model == 'gcn' or args.model == 'gat':
    n_conv_layers = args.n_conv_layers
else:
    n_conv_layers = 0

hyperparameter_defaults = dict(
    graph_name=GRAPH_NAMES[0],
    model='gcn',
    epochs=5000,
    learning_rate=5e-4,
    weight_decay=0,
    n_conv_layers=1,
    layer_sizes=[364, 364, 512, 256, 1]
)

torch.manual_seed(99)
np.random.seed(0)

wandb.init(project="brain-age-gnn", config=hyperparameter_defaults)

population_graph = preprocess.load_population_graph(graph_root, graph_name)
fold = evaluate.get_stratified_subject_split(population_graph)
evaluate.set_training_masks(population_graph, *fold)
preprocess.graph_feature_transform(population_graph)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# TODO correctly parse layer sizes
brain_gcn.gcn_train(population_graph, device,
                    n_conv_layers=n_conv_layers,
                    layer_sizes=[364, 364, 512, 256, 1],
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    epochs=args.epochs)
