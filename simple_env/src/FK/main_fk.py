#!/usr/bin/env python3

import argparse

# Customized modules
from data_synth import synthesize_dataset
from move_obs_embed import move_obstacle_embeddings

def main(args):
    '''
    Main function for performing forward kinematics
    '''

    
    # Synthesize data
    synthesize_dataset(args.raw_dataset_path, args.WS_names, args.ws_num_paths, args.synth_dataset_path, args.synth_ws_names)
    print('Done!')

    # Move obstacle embeddings
    move_obstacle_embeddings(args.raw_dataset_path, args.WS_names, args.synth_dataset_path, args.synth_ws_names)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Raw dataset related variables
    parser.add_argument('--raw_dataset_path', type=str, default='/home/davood/catkin_ws/src/GNN2/src/dataset', help='Raw dataset directory')
    parser.add_argument('--num_workspaces', type=int, default=17, help='Number of workspaces')
    parser.add_argument('--WS_names', nargs='+', type=str, default=['WS_1', 'WS_2', 'WS_3', 'WS_4', 'WS_5', 'WS_6',
                                                                    'WS_7', 'WS_8', 'WS_9', 'WS_10', 'WS_11', 'WS_12',
                                                                    'WS_13', 'WS_14', 'WS_15', 'WS_16', 'WS_17'], help='Name of the workspaces')
    parser.add_argument('--ws_num_paths', type=str, default={'WS_1': 425, 'WS_2': 952, 'WS_3': 889, 'WS_4': 645, 'WS_5': 488,
                                                             'WS_6': 745, 'WS_7': 541, 'WS_8': 537, 'WS_9': 963, 'WS_10': 964,
                                                             'WS_11': 615, 'WS_12': 692, 'WS_13': 647, 'WS_14': 523, 'WS_15': 516,
                                                             'WS_16': 402, 'WS_17': 547}, help='workspaces name and number of paths in them')

    # Synthesized dataset related variables
    parser.add_argument('--synth_dataset_path', type=str, default='/home/davood/catkin_ws/src/GNN2/src/dataset_FK1', help='Synthesized dataset directory.')
    # parser.add_argument('--synth_dataset_path', type=str, default='/home/davood/catkin_ws/src/GNN2/src/dataset_FK', help='Synthesized dataset directory.')
    parser.add_argument('--synth_ws_names', nargs='+', type=str, default=['WS_1', 'WS_2', 'WS_3', 'WS_4', 'WS_5', 'WS_6',
                                                                    'WS_7', 'WS_8', 'WS_9', 'WS_10', 'WS_11', 'WS_12',
                                                                    'WS_13', 'WS_14', 'WS_15', 'WS_16', 'WS_17'], help='Name of the workspaces of the synthesized data')

    args = parser.parse_args()
    main(args)
