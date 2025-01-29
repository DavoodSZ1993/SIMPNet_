#!/usr/bin/env python3

import os
import pickle 
import math
import numpy as np 

# Custom Modules
from collision_checking import IsInCollision
from graph_constructor import GraphConstructor
from planning_utils import trajectory_test, save_paths, load_model, load_normalization_info


class OnlinePlanning:
    def __init__(self, env_path:str, known_environment: bool, dof, step_size):

        self.env_path = env_path
        self.dof = dof
        self.step_size = step_size

        if known_environment:
            self.env_path = os.path.join(self.env_path, 'known_environment1')
        else:
            self.env_path = os.path.join(self.env_path, 'unknown_environment1')

    def load_obstacles(self):
        '''
        Loads obstacle embeddings for graph construction
        '''
        try: 
            filename = os.path.join(self.env_path, 'obs_embeddings.pkl')
            with open(filename, 'rb') as file:
                obs_embeddings = pickle.load(file)
                return obs_embeddings

        except FileNotFoundError as e:
            print('File cannot be found: obs_embeddings.pkl')

        except ValueError as e:
            print('File cannot be read: obs_embeddings.pkl')

        return None
        
    
    
    def load_start_goal_config(self, sample_idx):
        '''
        Loads start and goal configuration.
        '''

        try:
            filename = os.path.join(self.env_path, f'config_{sample_idx}.pkl')
            with open(filename, 'rb') as file:
                config = pickle.load(file)
                start = list(config['start config'])
                goal = list(config['end config'])
                return start, goal

        except FileNotFoundError as e:
            print(f'File cannot be found: config_{sample_idx}.pkl')
        
        except ValueError as e:
            print(f'File cannot be read: config_{sample_idx}.pkl')

        return None

    def steer(self, start, end, step_size, dof):
        '''
        Simple and direcr steering
        '''

        discretization_step = step_size
        dists = np.zeros(dof, dtype=np.float32)

        for i in range(dof):
            dists[i] = end[i] - start[i]

        dist_total = 0.0
        for i in range(dof):
            dist_total = dist_total + dists[i]*dists[i]
        dist_total = math.sqrt(dist_total)

        if dist_total > 0:
            increment_total = dist_total / discretization_step
            for i in range(dof):
                dists[i] = dists[i]/increment_total

            num_segments = int(math.floor(increment_total))
            state_curr = np.zeros(dof, dtype=np.float32)

            for i in range(dof):
                state_curr[i] = start[i]

            for i in range(num_segments):
                collision = IsInCollision(state_curr)
                if collision:
                    return False

                for j in range(dof):
                    state_curr[j] = state_curr[j] + dists[j]
            
            collision= IsInCollision(end)
            if collision:
                return False

        return True 

    def wrap_angle(self, theta):
        return [(t + np.pi) % (2 * np.pi) - np.pi for t in theta]

    def check_feasibility_entire_path(self, path):
        '''
        Checks the feasibility of entire path including the path edges
        '''

        for i in range(0, len(path)-1):
            ind= self.steer(path[i], path[i+1], self.step_size, self.dof)
            if not ind:
                return False

        return True

    def node_only_feasible(self, path):
        '''
        Checks the feasibility of path nodes only
        '''

        for i in range(0, len(path)):
            if IsInCollision(path[i]):
                return False
        return True

    def is_target_reached(self, start1, start2, dof=6):
        '''
        Criterion for checking when target is reached.
        '''

        s1 = np.zeros(dof, dtype=np.float32)
        for i in range(dof):
            s1[i] = start1[i]

        s2 = np.zeros(dof, dtype=np.float32)
        for i in range(dof):
            s2[i] = start2[i]

        for i in range(0, dof):
            if abs(s1[i] - s2[i]) > 0.05:
                return False
        return True

    def lazy_vertex_contraction(self, path, step_size):
        '''
        Lazy vertex contraction
        This function assumes that we cannot steer directly from the start of the path to the end. Otherwise we would have not been here.
        This is a MUST. Because we have dropout, and the samples are all over the place.
        Multiple iterations are required. Just ask ChatGPT to give some examples.
        '''
        for i in range(0, len(path) - 1):
            for j in range(len(path)-1, i+1, -1):
                ind = False
                ind= self.steer(path[i], path[j], self.step_size, self.dof)
                if ind == True:
                    pc = []
                    for k in range(0, i+1):
                        pc.append(path[k])
                    for k in range(j, len(path)):
                        pc.append(path[k])
                    return self.lazy_vertex_contraction(pc, step_size)
        return path
    
    def replan_path(self, model, known_environment, config_save_path, gaussian_normalization, p, goal, step_size):
        '''
        Re-planning in case the first planning step failed.
        '''

        step = 0
        path = []
        path.append(p[0])
        for i in range(1, len(p) - 1):
            collision= IsInCollision(p[i])
            if not collision:
                path.append(p[i])
        path.append(goal)

        new_path = []
        for i in range(0, len(path) - 1):
            target_reached = False 

            st = path[i]
            gl = path[i+1]
            steer = self.steer(st, gl, self.step_size, self.dof)

            if steer:
                new_path.append(st)
                new_path.append(gl)
            else:
                itr = 0 
                pA = []
                pA.append(st)
                pB = [] 
                pB.append(gl)
                target_reached = False 
                tree = 0 
                while (not target_reached) and itr < 3000:
                    itr = itr + 1
                    if tree == 0:
                        mlp_fw = MLPDataProcess(config_save_path, known_environment, st, gl)
                        ip1 = mlp_fw.mlp_input()

                        eps = 1e-5
                        if gaussian_normalization:
                            mean, std = load_normalization_info(gaussian_normalization)
                        ip1 = (ip1 - mean) / (std + eps)

                        model.eval()
                        st = model(ip1)
                        st = st.reshape(6, 2)
                        st = torch.atan2(st[:, 0], st[:, 1])
                        st = st.detach().squeeze().tolist()
                        pA.append(st)
                        tree = 1
                    else:
                        mlp_bw = MLPDataProcess(config_save_path, known_environment, gl, st)
                        ip2 = mlp_bw.mlp_input()

                        eps = 1e-5
                        if gaussian_normalization:
                            mean, std = load_normalization_info(gaussian_normalization)
                        ip2 = (ip2 - mean) / (sted + eps)

                        model.eval()
                        gl = model(ip2)
                        gl = gl.reshape(6, 2)
                        gl = torch.atan2(gl[:, 0], gl[:, 1])
                        gl = gl.detach().squeeze().tolist()
                        pB.append(gl)
                        tree = 0
                    target_reached = self.steer(st, gl, self.step_size, self.dof)
                if not target_reached:
                    print('Failed to replan')
                    return 0

                else:
                    for p1 in range(0, len(pA)):
                        new_path.append(pA[p1])
                    for p2 in range(len(pB)-1, -1, -1):
                        new_path.append(pB[p2])
        return new_path

    def compute_cost(self, path, dof):
        '''
        Computes the cost for each path.
        '''

        path = np.array(path)
        state_dists = []
        for i in range(len(path) - 1):
            dist = 0 
            for j in range(dof):
                diff = self.wrap_single_angle(path[i][j]) - self.wrap_single_angle(path[i+1][j])
                dist = dist + diff*diff 
            state_dists.append(np.sqrt(dist))
        total_cost = sum(state_dists)
        return total_cost

    def wrap_single_angle(self, theta):
        '''
        Wrap between -pi and pi for each angle
        '''
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        return theta
