"""
    Graph convolutional network implementation.
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

"""
import preprocess
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv

graph_root = 'data/graph'
graph_name = 'population_graph100.pt'
population_graph = preprocess.load_population_graph(graph_root, graph_name)


class BrainGCN(torch.nn.Module):
    def __init__(self):
        super(BrainGCN, self).__init__()
        self.conv1 = GCNConv(population_graph.num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.fc_1 = Linear(64, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.fc_1(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BrainGCN().to(device)
data = population_graph.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(10):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
# _, pred = model(data).max(dim=1)
# correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
# acc = correct / data.test_mask.sum().item()
