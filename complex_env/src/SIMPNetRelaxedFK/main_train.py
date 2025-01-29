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
    parser.add_argument('--datset_path', type=str, default='/home/davood/catkin_ws/src/gnn4/src/dataset_FK', help='Dataset directory')
    parser.add_argument('--label_dataset_path', type=str, default='/home/davood/catkin_ws/src/gnn4/src/dataset', help='Path of labels for training')
    parser.add_argument('--num_workers', type=int, default=11, help='Number of workspaces')
    parser.add_argument('--WS_names', nargs='+', type=str, default=['WS_1', 'WS_2', 'WS_3', 'WS_4', 'WS_5', 'WS_6',
                                                                    'WS_7', 'WS_8', 'WS_9', 'WS_10', 'WS_11'], help='Name of the workspaces')
    parser.add_argument('--ws_num_paths', type=str, default={'WS_1': 1000, 'WS_2': 1000, 'WS_3': 1000, 'WS_4': 1000, 'WS_5': 1000,
                                                             'WS_6': 478, 'WS_7': 1000, 'WS_8': 1000, 'WS_9': 680, 'WS_10': 1000,
                                                             'WS_11': 1000}, help='workspaces name and number of paths in them')
    parser.add_argument('--num_joints', type=int, default=6, help='Number of robot joints')
    parser.add_argument('--gaussian_normalization', type=bool, default=True, help='Determines the type of the normalization.')

    # Model Parameters
    parser.add_argument('--node_feature_size', type=int, default=6, help="Each node feature size.")
    parser.add_argument('--obs_feature_size', type=int, default=42, help="Size of obstacle embeddings for each environment.")
    parser.add_argument('--output_size', type=int, default=2, help="The size of output for each joint.")

    # Trainer parameters.
    parser.add_argument('--initial_lr', type=float, default=3e-4, help="Initial learning rate.")                           # Best so far: 5e-4
    parser.add_argument('--weight_decay', type=float, default=0.0, help="Weight decay for the optimizer.")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of epochs for training the planner.")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for training the neural network.")

    args = parser.parse_args()
    main(args)
