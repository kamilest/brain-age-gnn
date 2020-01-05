import numpy as np
import os

import sklearn

import preprocess
import precompute

graph_root = 'data/graph'
graph_name = 'population_graph_all_structural_euler_no_edges_sex.pt'
population_graph = preprocess.load_population_graph(graph_root, graph_name)

X_train = population_graph.x[population_graph.train_mask].cpu().detach().numpy()
X_validate = population_graph.x[population_graph.validate_mask].cpu().detach().numpy()

y_train = population_graph.y[population_graph.train_mask].cpu().detach().numpy().flatten()
y_validate = population_graph.y[population_graph.validate_mask].cpu().detach().numpy().flatten()

elastic_net = sklearn.linear_model.ElasticNet(l1_ratio=0.01, random_state=42)
elastic_net.fit(X_train, y_train)
print(elastic_net.score(X_validate, y_validate))

