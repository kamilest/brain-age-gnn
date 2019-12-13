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
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold

from torch.utils.tensorboard import SummaryWriter

graph_root = 'data/graph'
graph_name = 'population_graph_1000_PCA.pt'
population_graph = preprocess.load_population_graph(graph_root, graph_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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


def gcn_train(data):
    writer = SummaryWriter(
        log_dir='runs/PCA_fc2_841_1024_512_1_tanh_epochs=250_lr=0.005_weight_decay=1e-5')

    model = BrainGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    model.train()
    for epoch in range(250):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
        writer.add_scalar('Train/MSE',
                          loss.item(),
                          epoch)
        writer.add_scalar('Train/R2',
                          r2_score(data.y[data.train_mask].cpu().detach().numpy(),
                                   out[data.train_mask].cpu().detach().numpy()),
                          epoch)
        writer.add_scalar('Validation/MSE',
                          F.mse_loss(out[data.validate_mask], data.y[data.validate_mask]).item(),
                          epoch)
        writer.add_scalar('Validation/R2',
                          r2_score(data.y[data.validate_mask].cpu().detach().numpy(),
                                   out[data.validate_mask].cpu().detach().numpy()),
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
        # self.conv1 = GCNConv(population_graph.num_node_features, 1024)
        self.fc_1 = Linear(population_graph.num_node_features, 1024)
        self.fc_2 = Linear(1024, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.fc_1(x)
        x = torch.tanh(x)
        x = self.fc_2(x)

        return x


# torch.manual_seed(0)
# np.random.seed(0)

r2, predicted, actual = gcn_train(data)
