#!/usr/bin/env python3

import torch 
from torch import nn 
from torch_geometric.nn import GCNConv, Sequential, global_add_pool, global_mean_pool
import torch.nn.functional as F 


class MotionGenerator(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, lr: float, add_self_loops:bool, normalize:bool)->None:
        '''
        Let's construct the GNN
        '''

        super().__init__()

        self.gnn_net = Sequential('x, edge_index', [
            (GCNConv(in_channels, 64, normalize=normalize), 'x, edge_index -> x'),
            nn.ReLU(), nn.Dropout(),
            (GCNConv(64, 64, normalize=normalize), 'x, edge_index -> x'),
            nn.ReLU(), nn.Dropout(),
            (GCNConv(64, 64, normalize=normalize), 'x, edge_index -> x'),
            nn.ReLU(), nn.Dropout(),
            (GCNConv(64, 64, normalize=normalize), 'x, edge_index -> x'),
            nn.ReLU(), nn.Dropout(),
            (GCNConv(64, 64, normalize=normalize), 'x, edge_index -> x'),
            nn.ReLU(), nn.Dropout(),
            (GCNConv(64, 64, normalize=normalize), 'x, edge_index -> x'),
            nn.ReLU(), nn.Dropout(),
            (GCNConv(64, 64, normalize=normalize), 'x, edge_index -> x'),
            nn.ReLU(), nn.Dropout(),
            (GCNConv(64, 64, normalize=normalize), 'x, edge_index -> x'),
            nn.ReLU(), nn.Dropout(),
            (GCNConv(64, 64, normalize=normalize), 'x, edge_index -> x'),
            nn.ReLU(), nn.Dropout(),
            (GCNConv(64, 64, normalize=normalize), 'x, edge_index -> x'),
            nn.ReLU()])

        self.dropout = nn.Dropout()
        self.linear = nn.LazyLinear(out_channels)

        self.optimizer = torch.optim.Adam([
            dict(params=self.gnn_net.parameters(), weight_decay=5e-4),
            dict(params=self.linear.parameters(), weight_decay=0)], lr=lr)

        self.loss_fn = nn.MSELoss(reduction='mean')
    
    def forward(self, data):
        '''
        Forward pass - isn't it obvious
        '''
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gnn_net(x, edge_index)
        x = self.dropout(x)

        # Average over the graph
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        return x


