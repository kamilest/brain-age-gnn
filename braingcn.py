"""
    Graph convolutional network implementation.
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

"""
import preprocess
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

graph_root = 'data/graph'
graph_name = 'population_graph1000.pt'
population_graph = preprocess.load_population_graph(graph_root, graph_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = population_graph.to(device)


def train_braingcn(data, folds=5):
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

        model = BrainGCN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

        print('Training fold {}'.format(fold))
        # train fold
        model.train()
        for epoch in range(150):
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out[train_index], data.y[train_index])
            loss.backward()
            optimizer.step()

        model.eval()
        final_model = model(data)
        predicted = final_model[test_index].cpu()
        actual = data.y[test_index].cpu()
        r2 = r2_score(actual.detach().numpy(), predicted.detach().numpy())
        cv_scores.append(r2)
        print('Fold {} r2 score: {}'.format(fold, r2))
        print('Fold {} MSE: {}\n'.format(fold, F.mse_loss(predicted, actual)))

    return np.mean(cv_scores)


class BrainGCN(torch.nn.Module):
    def __init__(self):
        super(BrainGCN, self).__init__()
        self.conv1 = GCNConv(population_graph.num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.fc_1 = Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc_1(x)

        return x


torch.manual_seed(0)
np.random.seed(0)

print('Mean training set r^2 score', train_braingcn(data))
