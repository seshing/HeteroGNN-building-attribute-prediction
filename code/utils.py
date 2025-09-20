import numpy as np
import torch

from torch.nn import Linear
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import torch_geometric.transforms as T


def fill_na_in_df(df):
    na_cols = [c for c in df.columns if df[c].isna().any()]
    for c in na_cols:
        df[c].fillna(df[c].mean(), inplace=True)
    return df


def to_pyg_graph(geo_store, edge_store, target_col='building', target_value=[]):
    data = HeteroData()
    objects_copy = geo_store.copy()
    node_types = ['boundary', 'plot', 'building', 'street', 'intersection']
    scalers = {nt: StandardScaler() for nt in node_types}
    
    for node in node_types:
        df = objects_copy[node].drop(columns=['geometry'], errors='ignore')
        df = df.fillna(0)               # or your preferred imputation
        arr = df.to_numpy().astype(np.float32)
        arr_scaled = scalers[node].fit_transform(arr)
        
        data[node].x = torch.from_numpy(arr_scaled)
    
    # … now add edges, labels, and any transforms …
    for key, arr in edge_store.items():
        splitted = key.split('_')
        if 'rev' in key:
            data[splitted[0], 'rev_to', splitted[-1]].edge_index = torch.from_numpy(arr).to(torch.int64)
        else:
            data[splitted[0], 'to', splitted[2]].edge_index = torch.from_numpy(arr.copy()).to(torch.int64)
    if target_value:
        data[target_col].y = torch.from_numpy(np.array(target_value))
    data = T.AddSelfLoops()(data)
    return data
    
class HeteroSAGEEmbed(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('intersection', 'to', 'street'): SAGEConv((-1, -1), hidden_channels),
            ('street', 'to', 'intersection'): SAGEConv((-1, -1), hidden_channels),  # reverse
            ('plot', 'to', 'building'): SAGEConv((-1, -1), hidden_channels),
            ('building', 'to', 'plot'): SAGEConv((-1, -1), hidden_channels),  # reverse
            ('plot', 'to', 'plot'): SAGEConv((-1, -1), hidden_channels),
            ('building', 'to', 'building'): SAGEConv((-1, -1), hidden_channels),
            ('street', 'to', 'plot'): SAGEConv((-1, -1), hidden_channels), 
            ('plot', 'to', 'street'): SAGEConv((-1, -1), hidden_channels),   # reverse
            ('street', 'to', 'building'): SAGEConv((-1, -1), hidden_channels),
            ('building', 'to', 'street'): SAGEConv((-1, -1), hidden_channels),     # reverse
        }, aggr='mean')
        
        self.conv2 = HeteroConv({
            ('plot', 'to', 'building'): SAGEConv((-1, -1), hidden_channels),
            ('building', 'to', 'plot'): SAGEConv((-1, -1), hidden_channels),
            ('street', 'to', 'building'): SAGEConv((-1, -1), hidden_channels),
            ('building', 'to', 'street'): SAGEConv((-1, -1), hidden_channels),
            ('building', 'to', 'building'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='mean')

        self.fc1 = Linear(hidden_channels, out_channels)

    def embed(self, x_dict, edge_index_dict):
        # exactly the same as forward, minus the final fc1
        x = self.conv1(x_dict, edge_index_dict)
        x = {k: F.relu(v) for k,v in x.items()}

        x = self.conv2(x, edge_index_dict)
        x = {k: F.relu(v) for k,v in x.items()}

        return x['building']   # <-- raw hidden features, before fc1

    def forward(self, x_dict, edge_index_dict):
        emb = self.embed(x_dict, edge_index_dict)
        return self.fc1(emb)
