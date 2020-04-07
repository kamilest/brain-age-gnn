"""
    Graph convolutional network implementation.
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

"""

import enum

import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv


class ConvTypes(enum.Enum):
    GCN = 'GCN'
    GAT = 'GAT'


class BrainGNN(torch.nn.Module):
    # noinspection PyUnresolvedReferences
    def __init__(self, conv_type, num_node_features, n_conv_layers, layer_sizes, dropout_p):
        super(BrainGNN, self).__init__()
        self.conv = torch.nn.ModuleList()
        self.fc = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()
        size = num_node_features
        self.params = torch.nn.ParameterList([size].extend(layer_sizes))
        for i in range(n_conv_layers):
            if conv_type == ConvTypes.GCN:
                self.conv.append(GCNConv(size, layer_sizes[i]))
            elif conv_type == ConvTypes.GAT:
                self.conv.append(GATConv(size, layer_sizes[i]))
            else:
                self.conv.append(Linear(size, layer_sizes[i]))
            size = layer_sizes[i]
        for i in range(len(layer_sizes) - n_conv_layers):
            self.fc.append(Linear(size, layer_sizes[n_conv_layers+i]))
            size = layer_sizes[n_conv_layers+i]
            if i < len(layer_sizes) - n_conv_layers - 1:
                self.dropout.append(torch.nn.Dropout(dropout_p))

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(len(self.conv)):
            x = self.conv[i](x, edge_index)
            x = torch.tanh(x)
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = torch.tanh(x)
            x = self.dropout[i](x)

        x = self.fc[-1](x)
        return x


class BrainGCN(BrainGNN):
    def __init__(self, num_node_features, n_conv_layers, layer_sizes, dropout_p):
        super(BrainGCN, self).__init__(ConvTypes.GCN, num_node_features, n_conv_layers, layer_sizes, dropout_p)


class BrainGAT(BrainGNN):
    def __init__(self, num_node_features, n_conv_layers, layer_sizes, dropout_p):
        super(BrainGAT, self).__init__(ConvTypes.GAT, num_node_features, n_conv_layers, layer_sizes, dropout_p)
