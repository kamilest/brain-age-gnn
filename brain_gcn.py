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
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from torch.nn import Linear
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import GCNConv

import preprocess

graph_root = 'data/graph'
graph_name = 'population_graph_all_SEX_FTE_FI_MEM_structural_euler.pt'
population_graph = preprocess.load_population_graph(graph_root, graph_name)

logdir = './runs/{}'.format(datetime.now().strftime('%Y-%m-%d'))
Path(logdir).mkdir(parents=True, exist_ok=True)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
data = population_graph.to(device)


def gcn_train_cv(data, folds=5):
    train_idx = np.argwhere(data.train_mask.cpu().numpy())

    X = data.x[train_idx].cpu().numpy()
    y = np.squeeze(data.y[train_idx].cpu().numpy(), axis=(2,))
    print(data.y[train_idx].shape)
    print(y.shape)

    skf = StratifiedKFold(n_splits=folds, random_state=0)
    skf.get_n_splits()

    cv_scores = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        train_index = np.take(train_idx, tr)
        test_index = np.take(train_idx, te)

        # assert(len(np.intersect1d(train_index, np.argwhere(data.test_mask.cpu().numpy()))) == 0)
        # assert(len(np.intersect1d(test_index, np.argwhere(data.test_mask.cpu().numpy()))) == 0)

        print('Training fold {}'.format(fold))
        cv_scores.append(gcn_train(data))

    return np.mean(cv_scores)


# TODO automatic layer parameteristaion through arrays of layer sizes
def gcn_train(data, log=True):
    epochs = 350
    lr = 0.005
    weight_decay = 1e-5
    writer = None

    if log:
        log_name = '{}_nogcn_fc3_{}_1024_512_256_1_tanh_epochs={}_lr={}_weight_decay={}_{}'.format(
            graph_name.replace('population_graph_', '').replace('.pt', ''),
            data.num_node_features, epochs, lr, weight_decay, datetime.now().strftime("_%H_%M_%S"))
        writer = SummaryWriter(log_dir=os.path.join(logdir, log_name))

    model = BrainGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        if log:
            writer.add_scalar('Train/MSE',
                              loss.item(),
                              epoch)
            writer.add_scalar('Train/R2',
                              r2_score(data.y[data.train_mask].cpu().detach().numpy(),
                                       out[data.train_mask].cpu().detach().numpy()),
                              epoch)
            writer.add_scalar('Train/R',
                              pearsonr(data.y[data.train_mask].cpu().detach().numpy().flatten(),
                                       out[data.train_mask].cpu().detach().numpy().flatten())[0],
                              epoch)
            writer.add_scalar('Validation/MSE',
                              F.mse_loss(out[data.validate_mask], data.y[data.validate_mask]).item(),
                              epoch)
            writer.add_scalar('Validation/R2',
                              r2_score(data.y[data.validate_mask].cpu().detach().numpy(),
                                       out[data.validate_mask].cpu().detach().numpy()),
                              epoch)
            writer.add_scalar('Validation/R',
                              pearsonr(data.y[data.validate_mask].cpu().detach().numpy().flatten(),
                                       out[data.validate_mask].cpu().detach().numpy().flatten())[0],
                              epoch)
        print(epoch,
              loss.item(),
              r2_score(data.y[data.train_mask].cpu().detach().numpy(),
                       out[data.train_mask].cpu().detach().numpy()),
              pearsonr(data.y[data.train_mask].cpu().detach().numpy().flatten(),
                       out[data.train_mask].cpu().detach().numpy().flatten()))
        print(epoch,
              F.mse_loss(out[data.validate_mask],
                         data.y[data.validate_mask]).item(),
              r2_score(data.y[data.validate_mask].cpu().detach().numpy(),
                       out[data.validate_mask].cpu().detach().numpy()),
              pearsonr(data.y[data.validate_mask].cpu().detach().numpy().flatten(),
                       out[data.validate_mask].cpu().detach().numpy().flatten()))
        print()

        loss.backward()
        optimizer.step()

    if log:
        writer.close()

    model.eval()
    final_model = model(data)
    predicted = final_model[data.validate_mask].cpu()
    actual = data.y[data.validate_mask].cpu()
    r2 = r2_score(actual.detach().numpy(), predicted.detach().numpy())
    r = pearsonr(actual.detach().numpy().flatten(), predicted.detach().numpy().flatten())
    print('Final validation r2: {}'.format(r2))
    print('Final validation MSE: {}'.format(F.mse_loss(predicted, actual)))
    print('Final Pearson\'s r: {}'.format(r))

    return r2, predicted, actual


class BrainGCN(torch.nn.Module):
    def __init__(self):
        super(BrainGCN, self).__init__()
        self.conv1 = GCNConv(population_graph.num_node_features, 1024)
        self.fc_0 = Linear(population_graph.num_node_features, 1024)
        self.fc_1 = Linear(1024, 512)
        self.fc_2 = Linear(512, 256)
        self.fc_3 = Linear(256, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = self.conv1(x, edge_index)
        x = self.fc_0(x)
        x = torch.tanh(x)
        x = self.fc_1(x)
        x = torch.tanh(x)
        x = self.fc_2(x)
        x = torch.tanh(x)
        x = self.fc_3(x)

        return x


# torch.manual_seed(0)
# np.random.seed(0)

r2, predicted, actual = gcn_train(data)
