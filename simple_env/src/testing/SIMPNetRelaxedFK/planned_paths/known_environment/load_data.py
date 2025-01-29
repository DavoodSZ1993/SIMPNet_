#!/usr/bin/env python3

import pickle
import numpy as np

def compute_cost(path, dof=6):
    '''
    Computes the cost for each path.
    '''

    path = np.array(path)
    state_dists = []
    for i in range(len(path) - 1):
        dist = 0 
        for j in range(dof):
            diff = path[i][j]- path[i+1][j]
            print(f'Diff is: {diff}')
            dist = dist + diff*diff 
        print(f'Dist is: {dist}')
        state_dists.append(np.sqrt(dist))
        print(f'State dists: {state_dists}')
    total_cost = sum(state_dists)
    return total_cost

with open('planned_path_1.pkl', 'rb') as file:
    path = pickle.load(file)

print(f'The path is: {path}')
cost = compute_cost(path)
print(f'The cost is: {cost}')

print(len(path))

new_path = []
new_path.append(path[0])
#new_path.append(path[3])
#new_path.append(path[5])
new_path.append(path[-1])

print(f'The new path is: {new_path}')
cost = compute_cost(new_path)
print(f'The cost is: {cost}')
