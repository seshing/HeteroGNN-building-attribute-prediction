import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HeteroSAGE(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, fusion_dim=512, fusion_p=0.2, dropout_rate=0.5):
        super().__init__()

        # 1) NEW: fuse the (semantic + satellite + SVI) vector on 'building'
        self.fuse_building = Linear(-1, fusion_dim)   # infers input dim at first forward
        self.fuse_p = fusion_p
        self.dropout_rate = dropout_rate

        self.conv1 = HeteroConv({
            ('intersection', 'to', 'street'):   SAGEConv((-1, -1), hidden_channels),
            ('street', 'to', 'intersection'):   SAGEConv((-1, -1), hidden_channels),
            ('plot', 'to', 'plot'):             SAGEConv((-1, -1), hidden_channels),
            ('building', 'to', 'building'):     SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('plot', 'to', 'building'):         SAGEConv((-1, -1), hidden_channels),
            ('building', 'to', 'building'):     SAGEConv((-1, -1), hidden_channels),
            ('street', 'to', 'building'):       SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')

        self.fc1 = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = x_dict.copy()
        x_dict['building'] = F.relu(self.fuse_building(x_dict['building']))
        x_dict['building'] = F.dropout(x_dict['building'], p=self.fuse_p, training=self.training)

        # Block 1
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.dropout(F.relu(v), p=self.dropout_rate, training=self.training) for k, v in x_dict.items()}

        # Block 2
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.dropout(F.relu(v), p=self.dropout_rate, training=self.training) for k, v in x_dict.items()}

        out = self.fc1(x_dict['building'])
        return out


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
