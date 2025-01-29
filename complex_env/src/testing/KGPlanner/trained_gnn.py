#!/usr/bin/env python3

import torch
import numpy as np 
from torch import nn 
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import GCNConv, Sequential, global_add_pool

class MessagePassingNet(nn.Module):
    def __init__(self, in_channels:int, out_channels: int, add_self_loops:bool = True, normalize:bool = True):
        '''
        Let's construct the GNN
        Note: Dropout won't work here - Loss will not get reduced!
        ReLU is a good choice!
        '''
        super().__init__()
        self.gnn_net = Sequential('x, edge_index', [
            (GCNConv(in_channels, out_channels, normalize=normalize), 'x, edge_index -> x'),
            nn.PReLU(),nn.Dropout()])

    def forward(self, data):
        '''
        Forward pass - Isn't it obvious?
        '''

        x, edge_index, batch = data.x, data.edge_index, data.batch


        x = self.gnn_net(x, edge_index)
        return x

class NodeEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        '''
        Embedding node features
        '''

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 32), nn.PReLU(), nn.Dropout(),
            nn.Linear(32, out_features))

    def forward(self, data):
        '''
        This is the forward pass
        '''
        out = self.mlp(data)
        return out


class ObstacleEmbedding(nn.Module):
    def __init__(self, in_features, out_features):
        '''
        Embedding for obstacle embeddings
        '''

        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 32), nn.PReLU(), nn.Dropout(),
            nn.Linear(32, out_features))

    def forward(self, data):
        '''
        This is the forward pass
        '''

        out = self.mlp(data)
        return out

class CrossAttention(nn.Module):
    def __init__(self, in_features_embed, in_features_obs):
        
        super().__init__()
        self.key_net = nn.Linear(in_features_obs, in_features_embed, bias=False)
        self.query_net = nn.Linear(in_features_embed, in_features_embed, bias=False)
        self.value_net = nn.Linear(in_features_obs, in_features_embed, bias=False)
        self.layer_norm = nn.LayerNorm(in_features_embed, eps=1e-6)            # Applies Layer Normalization over a mini-batch of inputs.

    def forward(self, node_feature, obs_feature):
        '''
        Attention forward pass
        '''

        obs_key = self.key_net(obs_feature)
        node_query = self.query_net(node_feature)
        obs_value = self.value_net(obs_feature)

        #print(f'Shape of obstacle key: {obs_key.shape}')
        #print(f'Shape of node query: {node_query.shape}')
        #print(f'Shape of obstacle value: {obs_value.shape}')

        obs_attention = (node_query @ obs_key.T)

        #print(f'The shape of obstacle attention is: {obs_attention.shape}')
        
        node_feature_new = obs_attention @ obs_value

        #print(f'The shape of the node_feature_new after cross-attention: {node_feature_new.shape}')

        return self.layer_norm(node_feature_new + node_feature)

class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        '''
        Feedforward Blcok for Transformer Block
        '''
        super().__init__()
        self.w_1 = nn.Linear(in_dim, hidden_dim) # Position-wise
        self.w_2 = nn.Linear(hidden_dim, in_dim) # Position-wise
        self.layer_norm = torch.nn.LayerNorm(in_dim, eps=1e-6)

    def forward(self, x):
        residual = x

        x = self.w_2(self.w_1(x).relu())
        x = x + residual

        x = self.layer_norm(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, in_features_embed, in_features_obs):
        '''
        This is the transformer block.
        '''

        super().__init__()
        self.attention = CrossAttention(in_features_embed, in_features_obs)
        print(f'Number of attention block network parameters: {sum(p.numel() for p in self.attention.parameters() if p.requires_grad)}')
        #self.node_feed = FeedForward(in_features_embed, in_features_embed)
        #print(f'Number of attention block feedforward network parameters: {sum(p.numel() for p in self.node_feed.parameters() if p.requires_grad)}')

    def forward(self, node_feature, obs_feature):

        node_feature = self.attention(node_feature, obs_feature)
        #node_feature = self.node_feed(node_feature)

        return node_feature

# This is the main motion planning Neural Network.
class MotionGenerator(nn.Module):
    def __init__(self, node_feature_size, obs_feature_size, output_size, batch_size, num_nodes):
        '''
        This is the main motion generation module.
        '''

        super().__init__()
        self.batch_size = batch_size
        self.num_nodes = num_nodes

        # Step I - Node feature embedding & Obstacle node embedding.
        self.node_embed = NodeEmbedding(node_feature_size, 32)
        self.obs_embed = ObstacleEmbedding(obs_feature_size, 32)
        print(f'Number of Node embedding network parameters: {sum(p.numel() for p in self.node_embed.parameters() if p.requires_grad)}')
        print(f'Number of Obstacle embedding network parameters: {sum(p.numel() for p in self.obs_embed.parameters() if p.requires_grad)}')

        # Step II Cross-Attention between node features and obstacle embeddings
        self.cross_attentions = nn.ModuleList([TransformerBlock(32 * self.num_nodes, 32) for _ in range(1)])
        print(f'Number of cross attention network parameters: {sum(p.numel() for p in self.cross_attentions.parameters() if p.requires_grad)}')

        # Step III - Message Passing and output embedding
        self.message_passing = MessagePassingNet(in_channels=32, out_channels=32)
        print(f'Number of cross Message passing network parameters: {sum(p.numel() for p in self.message_passing.parameters() if p.requires_grad)}')

        # Step IV - Final embedding - getting a unique value for each joint.
        self.final_linear = nn.Linear(32, output_size)
        print(f'Number of Final layer network parameters: {sum(p.numel() for p in self.final_linear.parameters() if p.requires_grad)}')

    def forward(self, graph, obstacles):
        '''
        Main forward pass for motion planning
        '''

        node_features, edge_index, batch = graph.x, graph.edge_index, graph.batch
        #print(f'Shape of the node embeddings: {node_features}')
        #print(f'Shape of the obstacle embeddings: {obstacles.shape}')
        #self.batch_size = int(len(batch) / self.num_nodes)
        self.batch_size = 1
        #print(f'Batch size is: {self.batch_size}')

        # Step I - Embedding Process
        embedded_node_features = self.node_embed(node_features)
        embedded_obs_features = self.obs_embed(obstacles)
        #print(f'Shape of the embedded node features: {embedded_node_features.shape}')
        #print(f'Shape of the embedded obstacle features: {embedded_obs_features.shape}')
        #print(f'Shape of the reshaped embedded node features for cross-attention: {embedded_node_features.reshape(self.batch_size, -1).shape}')
        embedded_node_features = embedded_node_features.reshape(self.batch_size, -1)

        # Step II - Cross-Attention
        for attn in self.cross_attentions:
            augmented_node_features = attn(embedded_node_features, embedded_obs_features)

        # Step III - Message Passing
        #print(f'Shape of the embedded node features after cross-attention: {augmented_node_features.shape}')
        graph.x = augmented_node_features.reshape(self.batch_size * self.num_nodes, -1)
        #print(f'Node features shape after cross-attention: {graph.x.shape}')

        graph = self.message_passing(graph)
        #print(f'The shape of the graph: {graph.shape}')

        # Step IV - Final embedding
        joint_angles = self.final_linear(graph)
        #print(f'The shape of the output of the NN: {joint_angles.shape}')
        return joint_angles


'''
# More details: https://pytorch-geometric.readthedocs.io/en/latest/tutorial/create_gnn.html
Just for cuture use.
class MPNN(MessagePassing):
    def __init__(self, node_feature, aggr: str="mean", batch_norm: bool = False, **kwargs):
        super().__init__()
        self.batch_norm = batch_norm 
        self.lin_0 = nn.Sequence(
            nn.Linear(node_feature * 5, node_feature), nn.ReLU(),
            nn.Linear(node_feature, node_feature))                # This can be used before message passing (in message method!)
        self.lin_1 = nn.Linear(node_feature*2, node_feature)      # This can be done after the messgae passing (In the update method!)
        self.bn = nn.BatchNorm1d(node_feature)

    def forward(self, x, edge_index):
        ''''''
        Forward pass for message passing
        ''''''
        out = self.propogate(edge_index, x=x)
        out = self.bn(out) if self.batch_norm else out 

        return self.lin_1(torch.cat((x, out), dim=-1)) # This also can be done in the update method, or can be done here!

    def message(self, features, x_i, x_j):
        ''''''
        Here the mlp transforms both the target node feature x_i and the relative source features
        for each edge in the graph.
        ''''''

        z = torch.cat([x_j - x_i, x_j, x_i, edge_attr], dim=-1)
        values = self.lin_0(z)
        return values 

    def __repr__(self):
        return f'{self.__class__.__name__}({self.channels}, dim={self.dim})'''