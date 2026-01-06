import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HeteroSAGE(nn.Module):
    def __init__(self, hidden_channels, out_channels, fusion_dim=512, fusion_p=0.2, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.encoder = nn.Sequential(
            Linear(-1, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(fusion_p),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
        )

        self.conv1 = HeteroConv({
            ('intersection', 'to', 'street'):   SAGEConv((-1, -1), hidden_channels),
            ('street', 'to', 'intersection'):   SAGEConv((-1, -1), hidden_channels),
            ('plot', 'to', 'plot'):             SAGEConv((-1, -1), hidden_channels),
            ('building', 'to', 'building'):     SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('street', 'to', 'building'):       SAGEConv((-1, -1), hidden_channels),
            ('plot', 'to', 'building'):         SAGEConv((-1, -1), hidden_channels),
            ('building', 'to', 'building'):     SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')

        self.bn1 = nn.ModuleDict({
            'street':       nn.BatchNorm1d(hidden_channels),
            'intersection': nn.BatchNorm1d(hidden_channels),
            'building':     nn.BatchNorm1d(hidden_channels),
            'plot':         nn.BatchNorm1d(hidden_channels),
        })
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.fc1 = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = x_dict.copy()
        x_dict['building'] = self.encoder(x_dict['building'])

        # Block 1: Update all node types
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.dropout(F.relu(self.bn1[k](v)), p=self.dropout_rate, training=self.training) for k, v in x_dict.items()}

        # Block 2: Aggregate into 'building'
        x_dict = self.conv2(x_dict, edge_index_dict)
        out = self.bn2(x_dict['building'])
        out = F.dropout(F.relu(out), p=self.dropout_rate, training=self.training)

        return self.fc1(out)


class HeteroSAGEFull(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.conv1 = HeteroConv({
            ('plot', 'to', 'plot'): SAGEConv((-1, -1), hidden_channels),
            ('building', 'to', 'building'): SAGEConv((-1, -1), hidden_channels),
            ('intersection', 'to', 'street'): SAGEConv((-1, -1), hidden_channels),
            ('street', 'to', 'intersection'): SAGEConv((-1, -1), hidden_channels),
            ('plot','to','building'): SAGEConv((-1, -1), hidden_channels),
            ('building','to','plot'): SAGEConv((-1, -1), hidden_channels),
            ('plot','to','street'): SAGEConv((-1, -1), hidden_channels),
            ('street','to','plot'): SAGEConv((-1, -1), hidden_channels),
            ('building','to','street'): SAGEConv((-1, -1), hidden_channels),
            ('street','to','building'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('plot', 'to', 'plot'): SAGEConv((-1, -1), hidden_channels),
            ('building', 'to', 'building'): SAGEConv((-1, -1), hidden_channels),
            ('intersection', 'to', 'street'): SAGEConv((-1, -1), hidden_channels),
            ('street', 'to', 'intersection'): SAGEConv((-1, -1), hidden_channels),
            ('plot','to','building'): SAGEConv((-1, -1), hidden_channels),
            ('building','to','plot'): SAGEConv((-1, -1), hidden_channels),
            ('plot','to','street'): SAGEConv((-1, -1), hidden_channels),
            ('street','to','plot'): SAGEConv((-1, -1), hidden_channels),
            ('building','to','street'): SAGEConv((-1, -1), hidden_channels),
            ('street','to','building'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')
        
        self.conv3 = HeteroConv({
            ('building', 'to', 'building'):     SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')

        self.fc1 = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # Block 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.dropout(F.relu(v), p=self.dropout_rate, training=self.training) for k, v in x_dict.items()}

        # Block 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.dropout(F.relu(v), p=self.dropout_rate, training=self.training) for k, v in x_dict.items()}
        
        # Block 3
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {k: F.dropout(F.relu(v), p=self.dropout_rate, training=self.training) for k, v in x_dict.items()}

        out = self.fc1(x_dict['building'])
        return out
