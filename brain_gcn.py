"""
    Graph convolutional network implementation.
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

"""
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.nn import Linear
from torch_geometric.nn import GCNConv

import gnn_train_evaluate
import graph_construct
import graph_transform

graph_root = 'data/graph'
graph_name = 'population_graph_all_SEX_FTE_FI_structural_euler.pt'

hyperparameter_defaults = dict(
    model='gcn',
    epochs=5000,
    learning_rate=5e-4,
    dropout=0,
    weight_decay=0,
    n_conv_layers=1,
    layer_sizes=[364, 364, 512, 256, 1]
)


def gcn_train(graph, device, n_conv_layers=0, layer_sizes=None, epochs=3500, lr=0.005, dropout_p=0, weight_decay=1e-5,
              log=True, early_stopping=True, patience=10, delta=0.005, cv=False, fold=0, run_name=None,
              min_epochs=1000):
    data = graph.to(device)
    assert n_conv_layers >= 0

    if layer_sizes is None:
        layer_sizes = []

    model = BrainGCN(graph.num_node_features, n_conv_layers, layer_sizes, dropout_p).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Saving (early sopping) checkpoints
    early_stopping_checkpoint = 'checkpoint.pt'

    # Initialise early stopping.
    early_stopping_count = 0
    early_stopping_min_val_loss = None
    early_stop = False

    # Initialise wandb log.
    if log:
        wandb.run.save()
        wandb.watch(model)
        wandb.config.update({
            "graph_name": graph.name,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": weight_decay})

        if not cv or run_name is None:
            run_name = wandb.run.name
        wandb.run.name = run_name + '-fold-{}'.format(fold)
        wandb.run.save()
        early_stopping_checkpoint = '{}_{}_state_dict.pt'.format(
            datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
            wandb.run.name)

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

        if early_stopping:
            if early_stopping_min_val_loss is None:
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, early_stopping_checkpoint))
                early_stopping_min_val_loss = val_mse
            elif val_mse > early_stopping_min_val_loss - delta and epoch > min_epochs:
                early_stopping_count += 1
                if early_stopping_count >= patience:
                    early_stop = True
            else:
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, early_stopping_checkpoint))
                if log:
                    wandb.save(early_stopping_checkpoint)
                early_stopping_min_val_loss = val_mse
                early_stopping_count = 0

        if log:
            wandb.log({"train_mse_fold_{}".format(fold): train_mse[0],
                       "train_r2_fold_{}".format(fold): train_r2,
                       "train_r_fold_{}".format(fold): train_r,
                       "validation_mse_fold_{}".format(fold): val_mse,
                       "validation_r2_fold_{}".format(fold): val_r2[0],
                       "validation_r_fold_{}".format(fold): val_r})
        print(epoch, train_mse, train_r2, train_r)
        print(epoch, val_mse, val_r2, val_r)
        print()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        if early_stop:
            break

    # Load the best early stopping model.
    if early_stopping:
        model.load_state_dict(torch.load(os.path.join(wandb.run.dir, early_stopping_checkpoint)))

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
        wandb.run.summary["final_validation_mse_fold_{}".format(fold)] = F.mse_loss(predicted, actual)
        wandb.run.summary["final_validation_r2_fold_{}".format(fold)] = final_r2
        wandb.run.summary["final_validation_r_fold_{}".format(fold)] = final_r

        # Save the entire model.
        # best_model_name = 'best_{}.pt'.format(wandb.run.name)
        # torch.save(model, os.path.join(wandb.run.dir, best_model_name))
        # wandb.save(best_model_name)
        # wandb.join()

    return run_name, predicted, actual


class BrainGCN(torch.nn.Module):
    # noinspection PyUnresolvedReferences
    def __init__(self, num_node_features, n_conv_layers, layer_sizes, dropout_p):
        super(BrainGCN, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.fc = torch.nn.ModuleList()
        self.dropout = torch.nn.Dropout(dropout_p)
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
            x = self.dropout(x)

        x = self.fc[-1](x)
        return x


def gcn_train_with_cross_validation(graph, device, n_folds=10, n_conv_layers=0, layer_sizes=None, epochs=350, lr=0.005,
                                    dropout_p=0, weight_decay=1e-5, log=True, early_stopping=True, patience=10,
                                    delta=0.005):

    folds = gnn_train_evaluate.get_cv_subject_split(graph, n_folds=n_folds)
    results = []
    run_name = None

    wandb.save("*.pt")
    wandb.init(project="brain-age-gnn", config=hyperparameter_defaults, reinit=True)

    for i, fold in enumerate(folds):
        gnn_train_evaluate.set_training_masks(graph, *fold)
        graph_transform.graph_feature_transform(graph)

        fold_result = gcn_train(graph, device, n_conv_layers=n_conv_layers, layer_sizes=layer_sizes, epochs=epochs,
                                lr=lr, dropout_p=dropout_p, weight_decay=weight_decay, log=log,
                                early_stopping=early_stopping, patience=patience, delta=delta, cv=True, fold=i,
                                run_name=run_name)
        run_name = fold_result[0]
        fold_scores = fold_result[1:]
        results.append(fold_scores)

    # Add cross-validation summaries to the last fold
    cv_mse = [F.mse_loss(predicted, actual) for predicted, actual in results]
    cv_r2 = [r2_score(actual.detach().numpy(), predicted.detach().numpy()) for predicted, actual in results]
    cv_r = [pearsonr(actual.detach().numpy().flatten(), predicted.detach().numpy().flatten())
            for predicted, actual in results]
    # wandb.run.summary["cv_validation_mses"] = cv_mse
    # wandb.run.summary["cv_validation_r2s"] = cv_r2
    # wandb.run.summary["cv_validation_rs"] = cv_r
    wandb.run.summary["cv_validation_average_mse"] = np.mean(cv_mse, axis=0)
    wandb.run.summary["cv_validation_average_r2"] = np.mean(cv_r2, axis=0)
    wandb.run.summary["cv_validation_average_r"] = np.mean(cv_r, axis=0)

    return results


if __name__ == "__main__":
    torch.manual_seed(99)
    np.random.seed(0)

    population_graph = graph_construct.load_population_graph(graph_root, graph_name)
    fold = gnn_train_evaluate.get_stratified_subject_split(population_graph)
    gnn_train_evaluate.set_training_masks(population_graph, *fold)
    graph_transform.graph_feature_transform(population_graph)

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    gcn_train(population_graph, device, n_conv_layers=0, layer_sizes=[360, 256, 128, 1], lr=5e-4, weight_decay=0,
              epochs=5000)
