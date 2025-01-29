#!/usr/bin/env python3 

import numpy as np 
import math 
import os
import pickle

from get_fk_offline import GetFK

class FKDataset():
    '''
    Main class to perform Forward Kinematics to get joint positions
    '''
    def __init__(self, dataset_path: str, ws_name:str, ws_num_paths:int, synth_path:str, synth_ws_name:str)->None:
        self.dataset_path = dataset_path
        self.synth_path = synth_path

        self.ws_name = ws_name
        self.synth_ws_name = synth_ws_name 

        self.ws_path = os.path.join(self.dataset_path, self.ws_name)
        print(self.ws_path)
        self.synth_ws_path = os.path.join(self.synth_path, self.synth_ws_name)
        if not os.path.exists(self.synth_ws_path):
            os.makedirs(self.synth_ws_path)

        self.ws_num_paths = ws_num_paths

    def forward_kinematics(self)->None:
        for i in range(self.ws_num_paths):
            try:
                with open(os.path.join(self.ws_path, f'RRTstar_raw_{i}.pkl'), 'rb') as file:
                    traj = pickle.load(file)
                self.fk = GetFK(traj)
                traj_pose = self.fk.run()
                with open(os.path.join(self.synth_ws_path, f'traj_FK_new_{i}.pkl'), 'wb') as file:
                    pickle.dump(traj_pose, file)
            except FileNotFoundError as e:
                print(f'File not found: RRTstar_raw_{i}.pkl')
            except ValueError:
                print(f'Cannot load: RRTstar_raw_{i}.pkl')

def synthesize_dataset(org_path:str, org_ws:list, org_ws_num_paths: dict, synth_path:str, synth_ws:list):
    '''
    Gets joint values, and performs Forward Kinemaitcs to get joint positions in the 
    motion planning frame.
    '''

    '''
    # WS01
    org_path = org_path
    print(org_path)
    org_ws01_name = org_ws[0]
    print(org_ws01_name)
    org_ws01_num_path = org_ws_num_paths[org_ws01_name]

    synth_path = synth_path
    synth_ws01_name = synth_ws[0]

    get_fk_ws01 = FKDataset(org_path, org_ws01_name, org_ws01_num_path, synth_path, synth_ws01_name)
    get_fk_ws01.forward_kinematics()

    # WS02
    org_path = org_path
    print(org_path)
    org_ws02_name = org_ws[1]
    print(org_ws02_name)
    org_ws02_num_path = org_ws_num_paths[org_ws02_name]

    synth_path = synth_path
    synth_ws02_name = synth_ws[1]

    get_fk_ws02 = FKDataset(org_path, org_ws02_name, org_ws02_num_path, synth_path, synth_ws02_name)
    get_fk_ws02.forward_kinematics()

    # WS03
    org_path = org_path
    print(org_path)
    org_ws03_name = org_ws[2]
    print(org_ws03_name)
    org_ws03_num_path = org_ws_num_paths[org_ws03_name]

    synth_path = synth_path
    synth_ws03_name = synth_ws[2]

    get_fk_ws03 = FKDataset(org_path, org_ws03_name, org_ws03_num_path, synth_path, synth_ws03_name)
    get_fk_ws03.forward_kinematics()

    # WS04
    org_path = org_path
    print(org_path)
    org_ws04_name = org_ws[3]
    print(org_ws04_name)
    org_ws04_num_path = org_ws_num_paths[org_ws04_name]

    synth_path = synth_path
    synth_ws04_name = synth_ws[3]

    get_fk_ws04 = FKDataset(org_path, org_ws04_name, org_ws04_num_path, synth_path, synth_ws04_name)
    get_fk_ws04.forward_kinematics() '''


    org_path = org_path
    synth_path = synth_path
    print(len(org_ws))
    for i in range(len(org_ws)):
        org_ws_name = org_ws[i]
        org_ws_num_path = org_ws_num_paths[org_ws_name]
        synth_ws_name = synth_ws[i]

        get_fk_ws = FKDataset(org_path, org_ws_name, org_ws_num_path, synth_path, synth_ws_name)
        get_fk_ws.forward_kinematics() 


        