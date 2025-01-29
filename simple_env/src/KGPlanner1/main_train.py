#!/usr/bin/env python3

import argparse

# Custom modules
from graph_constructor import CustomDataset
from mp_gnn import MotionGenerator
from trainer import Trainer

def main(args):
    
    # Graph generator
    custom_dataset = CustomDataset(args.dataset_path, args.num_workers, args.WS_names,
                      args.ws_num_paths, args.num_joints, args.num_goals, args.num_obstacle_nodes, args.gaussian_normalization)
    dataset = custom_dataset.return_dataset()
    print(f'Number of graphs in the dataset: {len(dataset)}')

    # Training and validation
    model = MotionGenerator(args.in_channels, args.out_channels, args.learning_rate, args.add_self_loop, args.normalize)
    trainer = Trainer(model, dataset, args.num_epochs, args.batch_size, args.gaussian_normalization)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset Parameters
    parser.add_argument('--dataset_path', type=str, default='/home/davood/catkin_ws/src/GNN2/src/dataset', help='Path of labels for training')
    parser.add_argument('--num_workers', type=int, default=17, help='Number of workspaces')
    parser.add_argument('--WS_names', nargs='+', type=str, default=['WS_1', 'WS_2', 'WS_3', 'WS_4', 'WS_5', 'WS_6',
                                                                    'WS_7', 'WS_8', 'WS_9', 'WS_10', 'WS_11', 'WS_12',
                                                                    'WS_13', 'WS_14', 'WS_15', 'WS_16', 'WS_17'], help='Name of the workspaces')
    parser.add_argument('--ws_num_paths', type=str, default={'WS_1': 425, 'WS_2': 952, 'WS_3': 889, 'WS_4': 645, 'WS_5': 488,
                                                             'WS_6': 745, 'WS_7': 541, 'WS_8': 537, 'WS_9': 963, 'WS_10': 964,
                                                             'WS_11': 615, 'WS_12': 692, 'WS_13': 647, 'WS_14': 523, 'WS_15': 516,
                                                             'WS_16': 402, 'WS_17': 547}, help='workspaces name and number of paths in them')
    parser.add_argument('--num_joints', type=int, default=6, help='Number of robot joints')
    parser.add_argument('--num_goals', type=int, default=6, help='Number of joint goals')
    parser.add_argument('--num_obstacle_nodes', type=int, default=16, help='Number of obstacle nodes')
    parser.add_argument('--gaussian_normalization', type=bool, default=True, help='Determines the type of normalization for data pre-processing.')

    # Training and validation
    parser.add_argument('--in_channels', type=int, default=3, help='Number of features for each node')
    parser.add_argument('--out_channels', type=int, default=12, help='Number of goals (6 robot joints)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training the GCN')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training the NN.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training NN.')

    # Model parameters
    parser.add_argument('--add_self_loop', type=bool, default=True, help='Whether network have self-loops')
    parser.add_argument('--normalize', type=bool, default=False, help='Normalize argument in GCNConv')


    args = parser.parse_args()
    main(args)
    
