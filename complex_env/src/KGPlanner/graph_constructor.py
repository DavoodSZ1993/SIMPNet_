#!/usr/bin/env python3

import torch 
import numpy as np 
import os
import pickle
from torch_geometric.data import Data 
from torch.utils.data import Dataset

PI = np.pi 
CIRCULAR_LIMITS = -PI, PI 

class CustomDataset(Dataset):
    def __init__(self, path:str, label_path: str, num_ws:int, ws_names:list, ws_num_paths:dict, gaussian_normalization,
                 num_joints:int = None)->None:
        '''
        Graph Generator
        '''
        self.dataset_path = path
        self.label_dataset_path = label_path
        self.num_ws = num_ws 
        self.ws_names = ws_names 
        self.ws_num_paths = ws_num_paths
        self.num_joints = num_joints
        self.obstacle_names = ['Monitor', 'Desktop', 'Screwdriver_box', 'Disassembly_container', 'Desktop I', 'Desktop II', 'Disassembly_container_II']
        self.all_features = None
        self.all_labels = None
        self.all_obs_embed = None

        self.graph_dataset = []
        self.obs_dataset = []

        self.edge_index = self.edge_index_generator(self.num_joints)
        '''self.data_pre_proccess()
        torch.save(self.all_features, 'all_features.pt')
        torch.save(self.all_labels, 'all_labels.pt')
        torch.save(self.all_obs_embed, 'all_obs_embed.pt')'''
        self.normalize(gaussian_normalization)
        self.graph_construction()

        # save tensors


        assert len(self.graph_dataset) == len(self.obs_dataset), "Datasets must be of the same size"

        self.data_sanity_check()

    def edge_index_generator(self, num_joints)->torch.tensor:
        '''
        Constructing graph adjacency matrix
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

    def data_pre_proccess(self):
        '''
        Generates feature matrix and label vector for the graph
        '''

        for i in range(self.num_ws):
            ws_name = self.ws_names[i]
            ws_path = os.path.join(self.dataset_path, ws_name)
            ws_label_path = os.path.join(self.label_dataset_path, ws_name)
            ws_num_paths = self.ws_num_paths[ws_name]

            ws_paths = []
            ws_label_paths = []
            ws_obs_embeddings = []

            try:
                with open(os.path.join(ws_path, 'obs_embeddings.pkl'), 'rb') as file:
                    ws_obs = pickle.load(file)
                for obs_name in self.obstacle_names:
                    obs_center = ws_obs[obs_name]['center']
                    obs_size = ws_obs[obs_name]['size']
                    ws_obs_embeddings.append(list(obs_center) + list(obs_size))

            except FileNotFoundError as e:
                print(f'File cannot be found: obs_embeddings.pkl for workspace {i+1}')

            except ValueError as e:
                print(f'File cannot be read: obs_embeddings.pkl for workspace {i+1}')

            for j in range(ws_num_paths):
                try:
                    with open(os.path.join(ws_path, f'traj_FK_new_{j}.pkl'), 'rb') as file:
                        path = pickle.load(file)
                    ws_paths.append(path)
            
                except FileNotFoundError as e:
                    print(f'File cannot be found: traj_FK_new_{j}.pkl')

                except ValueError as e:
                    print(f'File cannot be read: traj_FK_new_{j}.pkl')

                try:
                    with open(os.path.join(ws_label_path, f'RRTstar_raw_{j}.pkl'), 'rb') as file:
                        path_label = pickle.load(file)
                    ws_label_paths.append(path_label)

                except FileNotFoundError as e:
                    print(f'File cannot be found: RRTstar_raw_{j}.pkl')

                except ValueError as e:
                    print(f'File cannot be read: RRTstar_raw_{j}.pkl')

            self.normalize_dataset(ws_paths, ws_label_paths, ws_obs_embeddings, ws_name)
    
    def ws_graph_constructor(self, ws_paths, ws_label_paths, ws_obs):
        '''
        Construct feature matrix, label vecotor, and obs embeddings for each workspace
        '''
        for j in range(len(ws_paths)):
            path = ws_paths[j]
            path_label = ws_label_paths[j]

            for i in range(len(path) - 1):
                node_embed = torch.tensor(path[i], dtype=torch.float32)
                goal_embed = torch.tensor(path[-1], dtype=torch.float32)
                diff_embed = torch.abs(node_embed - goal_embed)
                dist_embed = torch.norm(diff_embed, dim=1).unsqueeze(1)
                node_embed_ang_sin = torch.sin(self.wrap_angle(torch.tensor(path_label[i], dtype=torch.float32)).unsqueeze(1))
                node_embed_ang_cos = torch.cos(self.wrap_angle(torch.tensor(path_label[i], dtype=torch.float32)).unsqueeze(1))
                feature_matrix = torch.cat((node_embed, goal_embed, diff_embed, dist_embed, node_embed_ang_sin, node_embed_ang_cos), dim=1)
                #label_embed = self.wrap_angle(torch.tensor(path_label[i+1], dtype=torch.float32)).unsqueeze(1)
                #data = Data(x=feature_matrix, edge_index=self.edge_index.t().contiguous(), y=label_embed)
                label_embed_sin = torch.sin(self.wrap_angle(torch.tensor(path_label[i+1], dtype=torch.float32)).unsqueeze(1))
                label_embed_cos = torch.cos(self.wrap_angle(torch.tensor(path_label[i+1], dtype=torch.float32)).unsqueeze(1))
                label_embed = torch.cat((label_embed_cos, label_embed_sin), dim=1)
                data = Data(x=feature_matrix, edge_index=self.edge_index.t().contiguous(), y=label_embed)
                self.graph_dataset.append(data)

                ws_obs_embeddings = torch.tensor(ws_obs, dtype=torch.float32).reshape(1, -1).squeeze()
                self.obs_dataset.append(ws_obs_embeddings)

    def normalize_dataset(self, ws_paths, ws_label_paths, ws_obs, ws_name):
        '''
        Normalize the dataset featurewise.
        '''
        for j in range(len(ws_paths)):
            path = ws_paths[j]
            path_label = ws_label_paths[j]

            for i in range(len(path) - 1):
                node_embed = torch.tensor(path[i], dtype=torch.float32)
                goal_embed = torch.tensor(path[-1], dtype=torch.float32)
                diff_embed = torch.abs(node_embed - goal_embed)
                dist_embed = torch.norm(diff_embed, dim=1).unsqueeze(1)
                node_embed_ang_sin = torch.sin(self.wrap_angle(torch.tensor(path_label[i], dtype=torch.float32)).unsqueeze(1))
                node_embed_ang_cos = torch.cos(self.wrap_angle(torch.tensor(path_label[i], dtype=torch.float32)).unsqueeze(1))
                goal_embed_ang_sin = torch.sin(self.wrap_angle(torch.tensor(path_label[-1], dtype=torch.float32)).unsqueeze(1))
                goal_embed_ang_cos = torch.cos(self.wrap_angle(torch.tensor(path_label[-1], dtype=torch.float32)).unsqueeze(1))
                dist_embed_ang_sin = torch.abs(node_embed_ang_sin - goal_embed_ang_sin)
                dist_embed_ang_cos = torch.abs(node_embed_ang_cos - goal_embed_ang_cos)
                feature_matrix = torch.cat((node_embed, goal_embed, diff_embed, dist_embed, node_embed_ang_sin, node_embed_ang_cos, goal_embed_ang_sin, goal_embed_ang_cos,
                                            dist_embed_ang_sin, dist_embed_ang_cos), dim=1)
                if (i == 0) and (j == 0) and (ws_name == 'WS_1'):
                    self.all_features = feature_matrix
                else:
                    self.all_features = torch.cat((self.all_features, feature_matrix), dim=0)
                
                label_embed_sin = torch.sin(self.wrap_angle(torch.tensor(path_label[i+1], dtype=torch.float32)).unsqueeze(1))
                label_embed_cos = torch.cos(self.wrap_angle(torch.tensor(path_label[i+1], dtype=torch.float32)).unsqueeze(1))
                label_embed = torch.cat((label_embed_sin, label_embed_cos), dim=1)
                if (i == 0) and (j == 0) and (ws_name == 'WS_1'):
                    self.all_labels = label_embed
                else:
                    self.all_labels = torch.cat((self.all_labels, label_embed), dim=0)

                ws_obs_embeddings = torch.tensor(ws_obs, dtype=torch.float32).reshape(1, -1)
                print(f'The shape of obstacle embeddings: {ws_obs_embeddings.shape}')
                if (i == 0) and (j == 0) and (ws_name == 'WS_1'):
                    self.all_obs_embed = ws_obs_embeddings
                else:
                    self.all_obs_embed = torch.cat((self.all_obs_embed, ws_obs_embeddings), dim=0)
                
    def normalize(self, gaussian_normalization):
        '''
        Lets normalize the dataset.
        '''

        self.all_features = torch.load('all_features.pt')
        self.all_labels = torch.load('all_labels.pt')
        self.all_obs_embed = torch.load('all_obs_embed.pt')

        eps = 1e-3

        if gaussian_normalization:
            features_normal_info = {}
            mean_all_features = torch.mean(self.all_features, dim=0)
            std_all_features = torch.std(self.all_features, dim=0)
            self.all_features = (self.all_features - mean_all_features) / (std_all_features + eps)
            print(f'Mean of featues after normalization: {torch.mean(self.all_features, dim=0)}')
            print(f'Std of featues after normalization: {torch.std(self.all_features, dim=0)}')
            features_normal_info['mean'] = mean_all_features
            features_normal_info['std'] = std_all_features

            '''
            labels_normal_info = {}
            mean_all_labels = torch.mean(self.all_labels, dim=0)
            std_all_labels = torch.std(self.all_labels, dim=0)
            self.all_labels = (self.all_labels - mean_all_labels) / (std_all_labels + eps)
            print(f'Mean of labels after normalization: {torch.mean(self.all_labels, dim=0)}')
            print(f'Std of featues after normalization: {torch.std(self.all_labels, dim=0)}')
            labels_normal_info['mean'] = mean_all_labels
            labels_normal_info['std'] = std_all_labels'''

            obs_normal_info = {}
            mean_all_obs_embed = torch.mean(self.all_obs_embed, dim=0)
            std_all_obs_embed = torch.std(self.all_obs_embed, dim=0)
            self.all_obs_embed = (self.all_obs_embed - mean_all_obs_embed) / (std_all_obs_embed + eps)
            print(f'Mean of Obstacles after normalization: {torch.mean(self.all_obs_embed, dim=0)}')
            print(f'Std of featues after normalization: {torch.std(self.all_obs_embed, dim=0)}')
            obs_normal_info['mean'] = mean_all_obs_embed
            obs_normal_info['std'] = std_all_obs_embed

            normalization_info = {}
            normalization_info['all_features'] = features_normal_info
            #normalization_info['all_labels'] = labels_normal_info
            normalization_info['all_obstacles'] = obs_normal_info

            file_name = 'gaussian_normalization_info.pkl'
            with open(file_name, 'wb') as file:
                pickle.dump(normalization_info, file)

        else:
            features_normal_info = {}
            max_all_features = torch.max(self.all_features, dim=0).values()
            min_all_features = torch.min(self.all_features, dim=0).values()
            self.all_features = (self.all_features - min_all_features) / (max_all_features - min_all_features + eps)
            features_normal_info['max'] = max_all_features
            features_normal_info['min'] = min_all_features

            obs_normal_info = {}
            max_all_obs_embed = torch.max(self.all_obs_embed, dim=0).values()
            min_all_obs_embed = torch.min(slef.all_obs_embed, dim=0).values()
            self.all_obs_embed = (torch.all_obs_embed - min_all_obs_embed) / (max_all_obs_embed - min_all_obs_embed + eps)
            obs_normal_info['max'] = max_all_obs_embed
            obs_normal_info['min'] = min_all_obs_embed

            normalization_info = {}
            normalization_info['all_features'] = features_normal_info
            normalization_info['all_obstacles'] = obs_normal_info

            file_name = 'maxmin_normalization_info.pkl'
            with open(file_name, 'wb') as file:
                pickle.dump(normalization_info, file)


    def graph_construction(self):
        '''
        Create the dataset from the normalized dataset.
        '''

        for i in range(self.all_obs_embed.shape[0]):
            self.obs_dataset.append(self.all_obs_embed[i, :].squeeze())
            feature_matrix = self.all_features[i*6:(i+1)*6, :]
            label_embed = self.all_labels[i*6:(i+1)*6,:]
            data = Data(x=feature_matrix, edge_index=self.edge_index.t().contiguous(), y=label_embed)
            self.graph_dataset.append(data)

    def __len__(self):
        return len(self.graph_dataset)

    def __getitem__(self, idx):
        '''
        Load items from both datasets
        '''
        graph = self.graph_dataset[idx]
        obs_embed = self.obs_dataset[idx]

        return graph, obs_embed
    
    def data_sanity_check(self)->None:
        '''
        Check whether graph is constructed correctly!
        '''

        print(f'The length of graph dataset: {len(self.graph_dataset)}')
        graph = self.graph_dataset[0]
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

        # print
        print(f'The length of obstacle dataset: {len(self.obs_dataset)}')
        obs = self.obs_dataset[0]
        print(f'A sample of obstacle embeddings: {obs}')

    
    def return_dataset(self)->list:
        '''
        A method to get the dataset in the main function
        '''
        return self.dataset

    def wrap_angle(self, theta):
        return (theta + np.pi) % (2*np.pi) - np.pi
