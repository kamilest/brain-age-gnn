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


# TODO cross-validate within the training set, not in the entire set.
def cross_validation_score(data, n_splits=5):
    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()

    skf = StratifiedKFold(n_splits=n_splits, random_state=0)
    skf.get_n_splits()

    cv_scores = []
    for train_index, test_index in skf.split(X, y):
        model = BrainGCN().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

        print('Training fold ', len(cv_scores)+1)
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
        print('r2', r2)
        print('mse', F.mse_loss(predicted, actual))

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = population_graph.to(device)

torch.manual_seed(0)
np.random.seed(0)

print('final mean r2 score', cross_validation_score(data))
