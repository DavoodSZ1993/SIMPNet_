#!/usr/bin/env python3

import torch 
import numpy as np 
import os
import pickle
from torch_geometric.data import Data 

PI = np.pi 
CIRCULAR_LIMITS = -PI, PI 

class CustomDataset():
    def __init__(self, path: str, num_ws:int, ws_names:list, ws_num_paths:dict,
                 num_joints:int, num_goals:int, num_obstacle_nodes:int, gaussian_normalization)->None:
        '''
        Graph Generator
        '''
        self.datset_path = path
        self.num_ws = num_ws 
        self.ws_names = ws_names 
        self.ws_num_paths = ws_num_paths
        self.num_joints = num_joints
        self.num_goals = num_goals 
        self.num_obstacle_nodes = num_obstacle_nodes
        self.obstacle_names = ['Monitor', 'Desktop', 'Screwdriver_box', 'Disassembly_container']
        self.nodes_per_obstacle = 4
        self.dataset = []
        '''
        self.feature_label_generator()
        torch.save(self.all_features, 'all_features.pt')
        torch.save(self.all_labels, 'all_labels.pt')'''

        self.normalize(gaussian_normalization)

        self.edge_index = self.edge_index_generator(self.num_joints, self.num_goals, self.num_obstacle_nodes)
        self.construct_graph()
        self.data_sanity_check()

    def edge_index_generator(self, num_joints:int, num_goals:int, num_obstacle_nodes:int)->torch.tensor:
        '''
        Constructing graph adjacency matrix
        '''

        n_nodes = num_joints + num_goals + num_obstacle_nodes
        edge_index = []

        # Undirected edges between robot joints
        for i in range(num_joints - 1):
            edge = [i, i+1]
            edge_index.append(edge)
            edge = [i+1, i]
            edge_index.append(edge)

        # Self-loops in the Robot Joints
        for i in range(num_joints):
            edge = [i, i]
            edge_index.append(edge)

        # Directed edges --> from goals to joint nodes
        for i in range(num_goals):
            for j in range(num_joints):
                edge = [i + num_joints, j]
                edge_index.append(edge)

        # Directed edges --> from obstacle nodes to joint nodes
        for i in range(num_obstacle_nodes):
            for j in range(num_joints):
                edge = [i + num_joints + num_goals, j]
                edge_index.append(edge)

        edge_index = torch.tensor(np.array(edge_index))
        edge_index = edge_index.type(torch.long)
        return edge_index

    def feature_label_generator(self)->None:
        '''
        Generates feature matrix and label vector for the graph
        '''

        for i in range(self.num_ws):
            ws_name = self.ws_names[i]
            ws_path = os.path.join(self.datset_path, ws_name)
            ws_num_paths = self.ws_num_paths[ws_name]
            ws_idx = i

            self.oracle_path_extractor(ws_name, ws_path, ws_num_paths, ws_idx)


    def oracle_path_extractor(self, ws_name:str, ws_path:str, ws_num_paths:int, ws_idx:int)->list:
        '''
        Extracts oracle paths and obstacle embeddings from each workspace
        '''
        ws_paths = []
        ws_obs_embeddings = []

        try:
            with open(os.path.join(ws_path, 'obs_embeddings.pkl'), 'rb') as file:
                ws_obs = pickle.load(file)
                for obs_name in self.obstacle_names:
                    obs_edges = ws_obs[obs_name]['edges']
                    for i in range(self.nodes_per_obstacle):
                        obs_edge = list(obs_edges[i])
                        ws_obs_embeddings.append(obs_edge)

        except FileNotFoundError as e:
            print(f'File cannot be found: obs_embeddings.pkl for workspace {i+1}')

        except ValueError as e:
            print(f'File cannot be read: obs_embeddings.pkl for workspace {i+1}')

        for i in range(ws_num_paths):
            try:
                with open(os.path.join(ws_path, f'RRTstar_raw_{i}.pkl'), 'rb') as file:
                    path = pickle.load(file)
                    ws_paths.append(path)
            
            except FileNotFoundError as e:
                print(f'File cannot be found: RRTstar_raw_{i}.pkl')

            except ValueError as e:
                print(f'File cannot be read: RRTstar_raw_{i}.pkl')

        #self.ws_graph_constructor(ws_paths, ws_label_paths, ws_obs_embeddings)
        self.modified_ws_graph_contructor(ws_name, ws_paths, ws_obs_embeddings)

    def ws_graph_constructor(self, ws_paths: list, ws_label_paths:list, ws_obs:list)->None:
        '''
        Construct feature matrix, label vector for each of the workspaces.
        '''
        for j in range(len(ws_paths)):
            path = ws_paths[j]
            path_label = ws_label_paths[j]

            for i in range(len(path) - 1):
                node_embed = torch.tensor(path[i], dtype=torch.float32)
                goal_embed = torch.tensor(path[-1], dtype=torch.float32)
                label_embed = self.wrap_angle(torch.tensor(path_label[i+1], dtype=torch.float32))
                #print(f'Target angle for each graph: {label_embed}')
                ws_embed = torch.tensor(ws_obs, dtype=torch.float32)
                feature_matrix = torch.cat((node_embed, goal_embed, ws_embed), dim=0)
                data = Data(x = feature_matrix, edge_index=self.edge_index.t().contiguous(),y=label_embed)
                self.dataset.append(data)

    def modified_ws_graph_contructor(self, ws_name, ws_paths, ws_obs):
        '''
        Normalize dataset before anything.
        '''

        for j in range(len(ws_paths)):
            path = ws_paths[j]

            for i in range(len(path) - 1):
                node_embed_sin = torch.sin(self.wrap_angle(torch.tensor(path[i], dtype=torch.float32))).unsqueeze(1)
                node_embed_cos = torch.cos(self.wrap_angle(torch.tensor(path[i], dtype=torch.float32))).unsqueeze(1)
                node_embed = torch.cat((node_embed_sin, node_embed_cos, torch.zeros(node_embed_sin.shape[0], 1)), dim=1)
                goal_embed_sin = torch.sin(self.wrap_angle(torch.tensor(path[-1], dtype=torch.float32))).unsqueeze(1)
                goal_embed_cos = torch.cos(self.wrap_angle(torch.tensor(path[-1], dtype=torch.float32))).unsqueeze(1)
                goal_embed = torch.cat((goal_embed_sin, goal_embed_cos, torch.zeros(goal_embed_sin.shape[0], 1)), dim=1)
                label_embed_sin = torch.sin(self.wrap_angle(torch.tensor(path[i+1], dtype=torch.float32))).unsqueeze(1)
                label_embed_cos = torch.cos(self.wrap_angle(torch.tensor(path[i+1], dtype=torch.float32))).unsqueeze(1)
                ws_embed = torch.tensor(ws_obs, dtype=torch.float32)
                feature = torch.cat((node_embed, goal_embed, ws_embed), dim=0)
                label = torch.cat((label_embed_sin, label_embed_cos), dim=1)
                
                if (ws_name == 'WS_1') and (i == 0) and (j == 0):
                    self.all_features = feature
                    self.all_labels = label
                else:
                    self.all_features = torch.cat((self.all_features, feature), dim=0)
                    self.all_labels = torch.cat((self.all_labels, label), dim=0)

        print(f'The shape of all features: {self.all_features.shape}')
        print(f'The shape of all labels: {self.all_labels.shape}')

    def normalize(self, gaussian_normalization):
        '''
        Normalize dataset
        '''

        self.all_features = torch.load('all_features.pt')
        self.all_labels = torch.load('all_labels.pt')
        eps = 1e-5

        if gaussian_normalization:
            features_normal_info = {}
            mean_all_features = torch.mean(self.all_features, dim=0)
            std_all_features = torch.std(self.all_features, dim=0)
            self.all_features = (self.all_features - mean_all_features) / (std_all_features + eps)
            print(f'Mean of featues after normalization: {torch.mean(self.all_features, dim=0)}')
            print(f'Std of featues after normalization: {torch.std(self.all_features, dim=0)}')
            features_normal_info['mean'] = mean_all_features
            features_normal_info['std'] = std_all_features

            normalization_info = {}
            normalization_info['all_features'] = features_normal_info

            filename = 'gaussian_normalization_info.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(normalization_info, file)

        else:
            features_normal_info = {}
            max_all_features = torch.max(self.all_features, dim=0).values()
            min_all_features = torch.min(self.all_features, dim=0).values()
            self.all_features = (self.all_features - min_all_features) / (max_all_features - min_all_features + eps)
            features_normal_info['max'] = max_all_features
            features_normal_info['min'] = min_all_features

            normalization_info = {}
            normalization_info['all_features'] = features_normal_info

            filename = 'maxmin_normalization_info.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(normalization_info, file)

    def construct_graph(self):
        '''
        Construct data based on the normalized data
        '''

        num_data = int(self.all_labels.shape[0] / 6)
        print(f'Number of data: {num_data}')

        for i in range(num_data):
            feature_matrix = self.all_features[i*28: (i+1)*28, :]
            label_vector = self.all_labels[i*6: (i+1)*6, :]
            data = Data(x = feature_matrix, edge_index=self.edge_index.t().contiguous(), y=label_vector)
            self.dataset.append(data)


    def data_sanity_check(self)->None:
        '''
        Check whether graph is constructed correctly!
        '''

        graph = self.dataset[0]
        print(f'Graph Keys: {graph.keys()}')

        for key, item in graph:
            print(f'{key} found in data')

        print(f"Does graph have edge attributes: {'edge_attr' in graph}")
        print(f'Graph number of nodes: {graph.num_nodes}')
        print(f'Graph number of edges: {graph.num_edges}')
        print(f'Graph number of features per node: {graph.num_node_features}')
        print(f'Whether graph has isolated nodes: {graph.has_isolated_nodes()}')
        print(f'Whether graph gas self loops: {graph.has_self_loops()}')
        print(f'whether graph is directed: {graph.is_directed()}')

    
    def return_dataset(self)->list:
        '''
        A method to get the dataset in the main function
        '''
        return self.dataset

    def wrap_angle(self, theta):
        return (theta + np.pi) % (2*np.pi) - np.pi