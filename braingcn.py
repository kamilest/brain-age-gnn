"""
    Graph convolutional network implementation.
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

"""
import preprocess
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv

from sklearn.metrics import r2_score

graph_root = 'data/graph'
graph_name = 'population_graph100.pt'
population_graph = preprocess.load_population_graph(graph_root, graph_name)


class BrainGCN(torch.nn.Module):
    def __init__(self):
        super(BrainGCN, self).__init__()
        self.conv1 = GCNConv(population_graph.num_node_features, 16)
        self.conv2 = GCNConv(16, 32)
        self.conv3 = GCNConv(32, 64)
        self.conv4 = GCNConv(64, 128)
        self.conv5 = GCNConv(128, 256)
        self.fc_1 = Linear(256, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.fc_1(x)

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BrainGCN().to(device)
data = population_graph.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.mse_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

print('final train loss', F.mse_loss(model(data)[data.train_mask], data.y[data.train_mask]))
model.eval()
final_model = model(data)
predicted = final_model[data.test_mask].cpu()
actual = data.y[data.test_mask].cpu()
r2 = r2_score(predicted.detach().numpy(), actual.detach().numpy())
print(r2)
mse = F.mse_loss(final_model[data.test_mask], data.y[data.test_mask])
print(mse)
