import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNNodeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNNodeClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_weight):
        # Graph Convolution Layer 1
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        # Graph Convolution Layer 2
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        # Fully Connected Layer
        x = self.fc(x)
        return torch.sigmoid(x)  # Sigmoid for binary classification