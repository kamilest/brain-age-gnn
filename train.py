import argparse

import numpy as np
import torch

import brain_gcn
import preprocess

graph_root = 'data/graph'
graph_name = 'population_graph_all_SEX_FTE_FI_MEM_structural_euler.pt'
population_graph = preprocess.load_population_graph(graph_root, graph_name)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
data = population_graph.to(device)

parser = argparse.ArgumentParser(description='Brain age graph neural network.')
parser.add_argument('--graph_name', default=graph_name, type=str, help='Name of the graph')
parser.add_argument('--model', default='gcn', type=str, help='Type of model (options: gcn, fc)')
parser.add_argument('--epochs', default=5000, type=int, help='Number of epochs (default 5000)')
parser.add_argument('--learning_rate', default=5e-4, type=float, help='Learning rate (default 5e-4)')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay (default 1e-5)')
parser.add_argument('--layer_sizes', default='[364, 364, 512, 256, 1]', type=str, help='Sizes of layers')

torch.manual_seed(99)
np.random.seed(0)

r2, predicted, actual = brain_gcn.gcn_train(data, n_conv_layers=2, layer_sizes=[364, 364, 512, 256, 1], epochs=5000)