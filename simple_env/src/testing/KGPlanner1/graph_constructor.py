#!/usr/bin/env python3
import os 
import pickle
import numpy as np 
import torch 
from torch_geometric.data import Data, DataLoader

class GraphConstructor:
    def __init__(self, path, known_environment, start, goal):
        '''
        Graph constructor for training
        '''

        self.path = path
        self.known_environment = known_environment
        self.start= start
        self.goal = goal

        self.obstacle_embeddings = self.load_obstacle_embeddings()
        self.feature_matrix = self.construct_feature_matrix()
        self.edge_index = self.construct_edge_index()

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
                    obstacle_embedding = obs_embeddings['Monitor']['edges'].tolist()
                    [obstacle_embeddings.append(embed) for embed in obstacle_embedding]
                    obstacle_embedding = obs_embeddings['Desktop']['edges'].tolist()
                    [obstacle_embeddings.append(embed) for embed in obstacle_embedding]
                    obstacle_embedding = obs_embeddings['Disassembly_container']['edges'].tolist()
                    [obstacle_embeddings.append(embed) for embed in obstacle_embedding]
                    obstacle_embedding = obs_embeddings['Screwdriver_box']['edges'].tolist()
                    [obstacle_embeddings.append(embed) for embed in obstacle_embedding]
            
            except FileNotFoundError as e:
                print('File cannot be found: obs_embbedings.pkl')
            
            except ValueError as e:
                print('File cannot be read: obs_embeddings.pkl')

        else:
            file_path = os.path.join(self.path, 'unknown_environment')
            try:
                with open(os.path.join(file_path, 'obs_embeddings.pkl'), 'rb') as file:
                    obs_embeddings = pickle.load(file)
                    obstacle_embedding = obs_embeddings['Monitor']['edges'].tolist()
                    [obstacle_embeddings.append(embed) for embed in obstacle_embedding]
                    obstacle_embedding = obs_embeddings['Desktop']['edges'].tolist()
                    [obstacle_embeddings.append(embed) for embed in obstacle_embedding]
                    obstacle_embedding = obs_embeddings['Disassembly_container']['edges'].tolist()
                    [obstacle_embeddings.append(embed) for embed in obstacle_embedding]
                    obstacle_embedding = obs_embeddings['Screwdriver_box']['edges'].tolist()
                    [obstacle_embeddings.append(embed) for embed in obstacle_embedding]
            
            except FileNotFoundError as e:
                print('File cannot be found: obs_embeddings.pkl')
            
            except ValueError as e:
                print('File cannot be read: obs_embeddings.pkl')

        return obstacle_embeddings

    def construct_feature_matrix(self)->None:
        '''
        Constructing graph's feature matrix
        '''
        node_embed_sin = torch.sin(self.wrap_angle(torch.tensor(self.start, dtype=torch.float32))).unsqueeze(1)
        node_embed_cos = torch.cos(self.wrap_angle(torch.tensor(self.start, dtype=torch.float32))).unsqueeze(1)
        node_embed = torch.cat((node_embed_sin, node_embed_cos, torch.zeros(node_embed_sin.shape[0], 1)), dim=1)
        goal_embed_sin = torch.sin(self.wrap_angle(torch.tensor(self.goal, dtype=torch.float32))).unsqueeze(1)
        goal_embed_cos = torch.cos(self.wrap_angle(torch.tensor(self.goal, dtype=torch.float32))).unsqueeze(1)
        goal_embed = torch.cat((goal_embed_sin, goal_embed_cos, torch.zeros(goal_embed_sin.shape[0], 1)), dim=1)
        obs_embeddings = torch.tensor(self.obstacle_embeddings, dtype=torch.float32)
        feature_matrix = torch.cat((node_embed, goal_embed, obs_embeddings), dim=0)
        return feature_matrix 

    def construct_edge_index(self):
        '''
        Constructing graph's Adjacency matrix
        '''

        n_obstacles, n_joints, n_goals = len(self.obstacle_embeddings), len(self.start), len(self.goal)
        n_nodes = n_obstacles + n_joints + n_goals 
        edge_index = []

        # Undirected edges between robot joints
        for i in range(n_joints - 1):
            edge = [i, i+1]
            edge_index.append(edge)
            edge = [i+1, i]
            edge_index.append(edge)

        # Self-loops in the robot joints 
        for i in range(n_joints):
            edge = [i, i]
            edge_index.append(edge)

        # Directed edges --> from goals to joint nodes 
        for i in range(n_goals):
            for j in range(n_joints):
                edge = [i + n_joints, j]
                edge_index.append(edge)

        # Directed edges --> from obstacle nodes to joint_nodes
        for i in range(n_obstacles):
            for j in range(n_joints):
                edge = [i + n_joints + n_goals, j]
                edge_index.append(edge)

        return edge_index

    def construct_graph(self):
        '''
        Graph constuction goes here!!
        '''

        feature_matrix = self.feature_matrix
        edge_index = torch.tensor(np.array(self.edge_index), dtype=torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(feature_matrix, edge_index=edge_index)
        return data

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



