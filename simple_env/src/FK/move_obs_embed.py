#!/usr/bin/env python3

import os 
import shutil

def move_obstacle_embeddings(org_dataset_path, org_WSs, target_dataset_path, target_WSs):
    '''
    Just simply moves the obstacle embeddings
    from the original dataset to the synthesized dataset. 
    '''

    for i in range(len(org_WSs)):

        obs_org_path = os.path.join(org_dataset_path, f'WS_{i+1}/obs_embeddings.pkl')
        obs_target_path = os.path.join(target_dataset_path, f'WS_{i+1}/obs_embeddings.pkl')

        try:
            shutil.copy2(obs_org_path, obs_target_path)
            print(f'The obstacle embeddings from dataset/WS_{i+1} just got copied to the dataset_FK/WS_{i+1}')
        except IOError as e:
            print(f'I think we have an error: {e}')



