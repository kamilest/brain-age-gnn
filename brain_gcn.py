"""
    Graph convolutional network implementation.
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

"""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.nn import Linear
from torch_geometric.nn import GCNConv

import evaluate
import preprocess

wandb.init(project="brain-age-gnn")

graph_root = 'data/graph'
graph_name = 'population_graph_all_SEX_FTE_FI_structural_euler.pt'


logdir = './runs/{}'.format(datetime.now().strftime('%Y-%m-%d'))
Path(logdir).mkdir(parents=True, exist_ok=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def gcn_train(graph, device, n_conv_layers=0, layer_sizes=None, epochs=350, lr=0.005, weight_decay=1e-5, log=True):
    data = graph.to(device)
    assert n_conv_layers >= 0

    if layer_sizes is None:
        layer_sizes = []

    model = BrainGCN(graph.num_node_features, n_conv_layers, layer_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if log:
        wandb.watch(model)
        wandb.config.update({
            "graph_name": graph.name,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay})
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        train_mse = loss.item(),
        train_r2 = r2_score(data.y[data.train_mask].cpu().detach().numpy(),
                            out[data.train_mask].cpu().detach().numpy())
        train_r = pearsonr(data.y[data.train_mask].cpu().detach().numpy().flatten(),
                           out[data.train_mask].cpu().detach().numpy().flatten())[0]
        val_mse = F.mse_loss(out[data.validate_mask], data.y[data.validate_mask]).item()
        val_r2 = r2_score(data.y[data.validate_mask].cpu().detach().numpy(),
                          out[data.validate_mask].cpu().detach().numpy()),
        val_r = pearsonr(data.y[data.validate_mask].cpu().detach().numpy().flatten(),
                         out[data.validate_mask].cpu().detach().numpy().flatten())[0]
        if log:
            wandb.log({"Train MSE": train_mse[0],
                       "Train r2": train_r2,
                       "Train r": train_r,
                       "Validation MSE": val_mse,
                       "Validation r2": val_r2[0],
                       "Validation r": val_r})
        print(epoch, train_mse, train_r2, train_r)
        print(epoch, val_mse, val_r2, val_r)
        print()

        loss.backward()
        optimizer.step()

    model.eval()
    final_model = model(data)
    predicted = final_model[data.validate_mask].cpu()
    actual = data.y[data.validate_mask].cpu()
    final_r2 = r2_score(actual.detach().numpy(), predicted.detach().numpy())
    final_r = pearsonr(actual.detach().numpy().flatten(), predicted.detach().numpy().flatten())
    print('Final validation r2: {}'.format(final_r2))
    print('Final validation MSE: {}'.format(F.mse_loss(predicted, actual)))
    print('Final Pearson\'s r: {}'.format(final_r))
    if log:
        wandb.run.summary["Final validation MSE"] = F.mse_loss(predicted, actual)
        wandb.run.summary["Final validation r2"] = final_r2
        wandb.run.summary["Final validation r"] = final_r

    log_name = '{}_{}_gcn{}_fc{}_{}_{}_epochs={}_lr={}_weight_decay={}'.format(
        datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
        graph_name.replace('population_graph_', ''),
        n_conv_layers,
        len(layer_sizes) - n_conv_layers,
        data.num_node_features,
        '_'.join(map(str, layer_sizes)),
        epochs,
        lr,
        weight_decay)
    if log:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, log_name))

    return final_r2, predicted, actual


class BrainGCN(torch.nn.Module):
    # noinspection PyUnresolvedReferences
    def __init__(self, num_node_features, n_conv_layers, layer_sizes):
        super(BrainGCN, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.fc = torch.nn.ModuleList()
        size = num_node_features
        self.params = torch.nn.ParameterList([size].extend(layer_sizes))
        for i in range(n_conv_layers):
            self.conv.append(GCNConv(size, layer_sizes[i]))
            size = layer_sizes[i]
        for i in range(len(layer_sizes) - n_conv_layers):
            self.fc.append(Linear(size, layer_sizes[n_conv_layers+i]))
            size = layer_sizes[n_conv_layers+i]

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.conv)):
            x = self.conv[i](x, edge_index)
            x = torch.tanh(x)
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = torch.tanh(x)

        x = self.fc[-1](x)
        return x


def gcn_train_with_cross_validation(graph, n_folds=10):
    folds = evaluate.get_cv_subject_split(graph, n_folds=n_folds)

    results = []

    for fold in folds:
        evaluate.set_training_masks(graph, *fold)
        preprocess.graph_feature_transform(graph)

        result = gcn_train(graph, device, n_conv_layers=2, layer_sizes=[364, 364, 512, 256, 1], epochs=500)
        results.append(result)

    return results


torch.manual_seed(99)
np.random.seed(0)

population_graph = preprocess.load_population_graph(graph_root, graph_name)
fold = evaluate.get_stratified_subject_split(population_graph)
evaluate.set_training_masks(population_graph, *fold)
preprocess.graph_feature_transform(population_graph)

gcn_train(population_graph, device, n_conv_layers=2, layer_sizes=[360, 360, 512, 256, 256, 1], lr=1e-4, weight_decay=0, epochs=5000)
