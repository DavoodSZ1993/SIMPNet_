#!/usr/bin/env python3

import os 
import pickle
import torch
import numpy as np
import torch
import random

# Contains util functions for online planning.

def trajectory_test(planned_path, move_group):
    '''
    Check whether the planned path is valid!
    '''

    for waypoint in planned_path:
        joint_goal = waypoint
        joint_goal = [float(point) for point in joint_goal]

        move_group.go(joint_goal, wait=True)

def save_paths(directory, trajectory, known_environment, idx):
    '''
    Save the planned paths.
    '''

    if known_environment:
        folder_path = os.path.join(directory, 'known_environment')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(os.path.join(folder_path, f'planned_path_{idx}.pkl'), 'wb') as file:
            pickle.dump(trajectory, file)
    else:
        folder_path = os.path.join(directory, 'unknown_environment')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(os.path.join(folder_path, f'planned_path_{idx}.pkl'), 'wb') as file:
            pickle.dump(trajectory, file)

def load_normalization_info(gaussian_normalization=True):
    '''
    Laods the normalization info for GCN framework
    '''
    path = '/home/davood/catkin_ws/src/GNN2/src/Main_MPNN1'

    if gaussian_normalization:

        filename = 'gaussian_normalization_info.pkl'
        with open(os.path.join(path, filename), 'rb') as file:
            gaussian_normalization_info = pickle.load(file)

        features_gauss_norm_mean = gaussian_normalization_info['all_features']['mean']
        features_gauss_norm_std = gaussian_normalization_info['all_features']['std']
        obs_gauss_norm_mean = gaussian_normalization_info['all_obstacles']['mean']
        obs_gauss_norm_std = gaussian_normalization_info['all_obstacles']['mean']
        return features_gauss_norm_mean, features_gauss_norm_std, obs_gauss_norm_mean, obs_gauss_norm_std
    else:
        return None

def load_model(model, path:str, name:str):
    '''
    Loads the model parameters.
    '''
    model_path = os.path.join(path, name)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    return model

def save_planning_info(directory, known_environment, planning_times, planning_costs, successful_paths, unsuccessful_paths):
    '''
    Let save planning info
    '''

    planning_info = {}
    planning_info['planning times'] = planning_times
    planning_info['planning time'] = np.mean(np.array(planning_times))
    planning_info['planning costs'] = planning_costs
    planning_info['planning cost'] = np.mean(np.array(planning_costs))
    planning_info['successful paths'] = successful_paths
    planning_info['unsuccessful paths'] = unsuccessful_paths

    if known_environment:
        folder_path = os.path.join(directory, 'known_environment')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(os.path.join(folder_path, f'path_planning_info.pkl'), 'wb') as file:
            pickle.dump(planning_info, file)
    else:
        folder_path = os.path.join(directory, 'unknown_environment')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(os.path.join(folder_path, f'path_planning_info.pkl'), 'wb') as file:
            pickle.dump(planning_info, file)

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)



