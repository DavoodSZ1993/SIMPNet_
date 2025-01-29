#!/usr/bin/env python3

import argparse

# Custom Modules
from graph_constructor import CustomDataset
from mp_gnn import MotionGenerator
from trainer import Trainer


def main(args):
    
    # Graph generator
    custom_dataset = CustomDataset(args.datset_path, args.label_dataset_path, args.num_workers, args.WS_names,
                      args.ws_num_paths, args.gaussian_normalization, args.num_joints)

    # Motion generation
    model = MotionGenerator(args.node_feature_size, args.obs_feature_size, args.output_size, args.batch_size, args.num_joints)
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of the parameters of the neural network: {total_parameters}')
    trainer = Trainer(custom_dataset, model, args.initial_lr, args.num_epochs, args.batch_size, args.weight_decay, args.gaussian_normalization)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Dataset Parameters
    # parser.add_argument('--datset_path', type=str, default='/home/davood/catkin_ws/src/GNN2/src/dataset_FK', help='Dataset directory')
    parser.add_argument('--datset_path', type=str, default='/home/davood/catkin_ws/src/GNN2/src/dataset_FK1', help='Dataset directory')
    parser.add_argument('--label_dataset_path', type=str, default='/home/davood/catkin_ws/src/GNN2/src/dataset', help='Path of labels for training')
    parser.add_argument('--num_workers', type=int, default=17, help='Number of workspaces')
    parser.add_argument('--WS_names', nargs='+', type=str, default=['WS_1', 'WS_2', 'WS_3', 'WS_4', 'WS_5', 'WS_6',
                                                                    'WS_7', 'WS_8', 'WS_9', 'WS_10', 'WS_11', 'WS_12',
                                                                    'WS_13', 'WS_14', 'WS_15', 'WS_16', 'WS_17'], help='Name of the workspaces')
    parser.add_argument('--ws_num_paths', type=str, default={'WS_1': 425, 'WS_2': 952, 'WS_3': 889, 'WS_4': 645, 'WS_5': 488,
                                                             'WS_6': 745, 'WS_7': 541, 'WS_8': 537, 'WS_9': 963, 'WS_10': 964,
                                                             'WS_11': 615, 'WS_12': 692, 'WS_13': 647, 'WS_14': 523, 'WS_15': 516,
                                                             'WS_16': 402, 'WS_17': 547}, help='workspaces name and number of paths in them')
    parser.add_argument('--num_joints', type=int, default=6, help='Number of robot joints')
    parser.add_argument('--gaussian_normalization', type=bool, default=True, help='Determines the type of the normalization.')

    # Model Parameters
    parser.add_argument('--node_feature_size', type=int, default=16, help="Each node feature size.")
    parser.add_argument('--obs_feature_size', type=int, default=24, help="Size of obstacle embeddings for each environment.")
    parser.add_argument('--output_size', type=int, default=2, help="The size of output for each joint.")

    # Trainer parameters.
    parser.add_argument('--initial_lr', type=float, default=3e-4, help="Initial learning rate.")                           # Best so far: 5e-4
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay for the optimizer.")
    parser.add_argument('--num_epochs', type=int, default=30, help="Number of epochs for training the planner.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training the neural network.")

    args = parser.parse_args()
    main(args)
