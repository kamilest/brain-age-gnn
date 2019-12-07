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

graph_root = 'data/graph'
graph_name = 'population_graph1000.pt'
population_graph = preprocess.load_population_graph(graph_root, graph_name)


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
model = BrainGCN().to(device)
data = population_graph.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

torch.manual_seed(0)
np.random.seed(0)

model.train()
for epoch in range(300):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    if epoch % 10 == 9:
        print(epoch, loss)
    loss.backward()
    optimizer.step()

model.eval()
final_model = model(data)
predicted = final_model[data.train_mask].cpu()
actual = data.y[data.train_mask].cpu()
r2 = r2_score(actual.detach().numpy(), predicted.detach().numpy())
print(r2)
mse = F.mse_loss(final_model[data.test_mask], data.y[data.test_mask])
print(mse)
