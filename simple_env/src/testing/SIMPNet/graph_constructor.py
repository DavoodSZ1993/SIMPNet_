#!/usr/bin/env python3
import os 
import pickle
import numpy as np 
import torch 
from torch_geometric.data import Data, DataLoader

class GraphConstructor:
    def __init__(self, path, known_environment, start_3d, goal_3d, start, goal, num_joints):
        '''
        Graph constructor for training
        '''

        self.path = path
        self.known_environment = known_environment
        self.start_3d = start_3d
        self.goal_3d = goal_3d
        self.start= start
        self.goal = goal
        self.num_joints = num_joints

        self.obstacle_embeddings = torch.tensor(self.load_obstacle_embeddings(), dtype=torch.float32).reshape(1, -1)
        self.feature_matrix = self.construct_feature_matrix()
        self.edge_index = self.construct_edge_index(self.num_joints)

    def load_obstacle_embeddings(self):
        '''
        Loads obstacle embeddings
        '''

        obstacle_embeddings = []
        if self.known_environment:
            file_path = os.path.join(self.path, 'known_environment')
            try:
                with open(os.path.join(file_path, 'obs_embeddings.pkl'), 'rb') as file:
                    obs_embeddings = pickle.load(file)
                monitor_center = obs_embeddings['Monitor']['center']
                monitor_size = obs_embeddings['Monitor']['size']
                obstacle_embeddings.append(monitor_center + monitor_size)
                desktop_center = obs_embeddings['Desktop']['center']
                desktop_size = obs_embeddings['Desktop']['size']
                obstacle_embeddings.append(desktop_center + desktop_size)
                screwdriver_center = obs_embeddings['Screwdriver_box']['center']
                screwdriver_size = obs_embeddings['Screwdriver_box']['size']
                obstacle_embeddings.append(screwdriver_center + screwdriver_size)
                disassembly_center = obs_embeddings['Disassembly_container']['center']
                disassembly_size = obs_embeddings['Disassembly_container']['size']
                obstacle_embeddings.append(disassembly_center + disassembly_size)
            
            except FileNotFoundError as e:
                print('File cannot be found: obs_embbedings.pkl')
            
            except ValueError as e:
                print('File cannot be read: obs_embeddings.pkl')

        else:
            file_path = os.path.join(self.path, 'unknown_environment')
            try:
                with open(os.path.join(file_path, 'obs_embeddings.pkl'), 'rb') as file:
                    obs_embeddings = pickle.load(file)
                monitor_center = obs_embeddings['Monitor']['center']
                monitor_size = obs_embeddings['Monitor']['size']
                obstacle_embeddings.append(monitor_center + monitor_size)
                desktop_center = obs_embeddings['Desktop']['center']
                desktop_size = obs_embeddings['Desktop']['size']
                obstacle_embeddings.append(desktop_center + desktop_size)
                screwdriver_center = obs_embeddings['Screwdriver_box']['center']
                screwdriver_size = obs_embeddings['Screwdriver_box']['size']
                obstacle_embeddings.append(screwdriver_center + screwdriver_size)
                disassembly_center = obs_embeddings['Disassembly_container']['center']
                disassembly_size = obs_embeddings['Disassembly_container']['size']
                obstacle_embeddings.append(disassembly_center + disassembly_size)
            
            
            except FileNotFoundError as e:
                print('File cannot be found: obs_embeddings.pkl')
            
            except ValueError as e:
                print('File cannot be read: obs_embeddings.pkl')

        return obstacle_embeddings

    def construct_feature_matrix(self)->None:
        '''
        Constructing graph's feature matrix
        '''
        node_embed = torch.tensor(self.start_3d, dtype=torch.float32)
        goal_embed = torch.tensor(self.goal_3d, dtype=torch.float32)
        diff_embed = torch.abs(node_embed - goal_embed)
        dist_embed = torch.norm(diff_embed, dim=1).unsqueeze(1)
        node_embed_ang_sin = torch.sin(self.wrap_angle(torch.tensor(self.start, dtype=torch.float32)).unsqueeze(1))
        node_embed_ang_cos = torch.cos(self.wrap_angle(torch.tensor(self.start, dtype=torch.float32)).unsqueeze(1))
        goal_embed_ang_sin = torch.sin(self.wrap_angle(torch.tensor(self.goal, dtype=torch.float32)).unsqueeze(1))
        goal_embed_ang_cos = torch.cos(self.wrap_angle(torch.tensor(self.goal, dtype=torch.float32)).unsqueeze(1))
        dist_embed_ang_sin = torch.abs(node_embed_ang_sin - goal_embed_ang_sin)
        dist_embed_ang_cos = torch.abs(node_embed_ang_cos - goal_embed_ang_cos)
        feature_matrix = torch.cat((node_embed, goal_embed, diff_embed, dist_embed, node_embed_ang_sin, node_embed_ang_cos, goal_embed_ang_sin, 
                                    goal_embed_ang_cos, dist_embed_ang_sin, dist_embed_ang_cos), dim=1)

        return feature_matrix

    def construct_edge_index(self, num_joints):
        '''
        Constructing graph's Adjacency matrix
        '''

        n_nodes = num_joints
        edge_index = []

        # base_joint
        for i in range(n_nodes):
            edge = [0, i]
            edge_index.append(edge)


        # 1st joint
        for i in range(n_nodes):
            edge = [1, i]
            edge_index.append(edge)

        # 2nd joint
        for i in range(1, n_nodes):
            edge = [2, i]
            edge_index.append(edge)

        # 3rd joint
        for i in range(2, n_nodes):
            edge = [3, i]
            edge_index.append(edge)

        # 4th joint
        for i in range(3, n_nodes):
            edge = [4, i]
            edge_index.append(edge)

        # 5th joint
        for i in range(4, n_nodes):
            edge = [5, i]
            edge_index.append(edge)

        edge_index = torch.tensor(np.array(edge_index))
        edge_index = edge_index.type(torch.long)

        return edge_index

    def construct_graph(self):
        '''
        Graph constuction goes here!!
        '''

        feature_matrix = self.feature_matrix
        edge_index = torch.tensor(np.array(self.edge_index), dtype=torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(feature_matrix, edge_index=edge_index)
        return data, self.obstacle_embeddings

    def graph_sanity_check(self, data):
        '''
        Check the graph properties to make sure.
        '''

        print(f'Graph keys: {data.keys()}')
        print(f'Graph nodes: {data["x"]}')

        for key, item in data:
            print(f'{key} found in data')

        print(f"Does graph have edge attributes: {'edge_attr' in data}")
        print(f'Graph number of nodes: {data.num_nodes}')
        print(f'Graph number of edges: {data.num_edges}')
        print(f'Graph number of features per node: {data.num_node_features}')
        print(f'Whether graph has isolated nodes: {data.has_isolated_nodes()}')
        print(f'Whether graph has self loops: {data.has_self_loops()}')
        print(f'Whether graph is directed: {data.is_directed()}')

    def wrap_angle(self, theta):
        return (theta + np.pi) % (2*np.pi) - np.pi



