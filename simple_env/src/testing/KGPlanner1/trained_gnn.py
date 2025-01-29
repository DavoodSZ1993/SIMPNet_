#!/usr/bin/env python3

import torch 
from torch import nn 
from torch_geometric.nn import GCNConv, Sequential, global_add_pool, global_mean_pool
import torch.nn.functional as F

'''
The network already has one dropout. So, it is safe to say that we can use it for evaluation.
But let add an dropout module to be more general.
'''

class EvalDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p 
    
    def forward(self, x):
        '''
        Only apply dropout in evaluation mode
        '''

        if not self.training:
            return F.dropout(x, self.p, False)
        return x

class MotionGenerator(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, add_self_loops:bool, normalize:bool)->None:
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
        self.eval_dropout = EvalDropout()
        self.linear = nn.LazyLinear(out_channels)

        self.loss_fn = nn.MSELoss(reduction='mean')
    
    def forward(self, data):
        '''
        Forward pass - isn't it obvious
        '''
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gnn_net(x, edge_index)
        x = self.dropout(x)
        x = self.eval_dropout(x)

        # Average over the graph
        x = global_mean_pool(x, batch)
        x = self.linear(x)
        print(f'The output is: {x}')
        return x


