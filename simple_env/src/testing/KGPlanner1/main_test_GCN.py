#!/usr/bin/env python3

import argparse
import sys 
import rospy 
import moveit_commander
import torch
import pdb                       # pdb.trace() for debugging - press q to quit the debugging mode.
import time 
import numpy as np

# Customized modules
from environment import LoadEnvironment
from gen_start_goal import StartGoalConfig
from online_planning import OnlinePlanning
from planning_utils import trajectory_test, save_paths, load_model, load_normalization_info, save_planning_info
from get_fk_offline import GetFK
from graph_constructor import GraphConstructor
from trained_gnn import MotionGenerator
from collision_checking import IsInCollision

# Import messages
import moveit_msgs.msg


def main(args):
    # Variables
    col_times = []

    # Set up MoveIt Environment
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('gnn_planner', anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    group_name = "ur5e_arm"
    move_group = moveit_commander.MoveGroupCommander(group_name)
    group_name = "gripper"
    hand_group = moveit_commander.MoveGroupCommander(group_name)

    display_trajectory_publisher = rospy.Publisher('move_group/display_planned_path',
                                                   moveit_msgs.msg.DisplayTrajectory,
                                                   queue_size=20)
    # Load environment and close gripper.
    environment = LoadEnvironment(robot, scene, hand_group, args.obstacle_path, args.num_workspaces, 
                                  args.WS_names, args.known_environment, args.config_save_path)
    
    # Load the trained GNN
    model = MotionGenerator(args.in_channels, args.out_channels, args.add_self_loop, args.normalize)
    model = load_model(model, args.trained_model_path, args.trained_model_name)
    
    '''
    # Randomly generate start and goal configuration.
    config_generation = StartGoalConfig(move_group, args.num_start_end, args.config_save_path, args.known_environment)'''

    # Online Planning
    planning_module = OnlinePlanning(args.config_save_path, args.known_environment, args.dof, args.step_size)
    planning_module.load_obstacles()
    planning_times = []
    planning_costs = []
    num_steps = 300 
    successful_paths = 0

    for i in range(args.num_sampled_start_end):
        start1, goal1 = planning_module.load_start_goal_config(i)
        goal2, start2 = planning_module.load_start_goal_config(i)

        path1 = []
        path1.append(start1)
        path2 = []
        path2.append(start2)
        path = []
        target_reached = False 
        step=0
        tree=0
        t_start = time.time()

        while (not target_reached) and step < num_steps:
            step = step + 1 

            if tree == 0:
                graph_const_fw = GraphConstructor(args.config_save_path, args.known_environment, start1, start2)
                graph_fw = graph_const_fw.construct_graph()

                eps = 1e-5
                if args.gaussian_normalization:
                    mean, std = load_normalization_info(args.gaussian_normalization)
                graph_fw['x'] = (graph_fw['x'] - mean) / (std + 1e-5)

                start1 = model(graph_fw)
                start1 = start1.reshape(6, 2)
                start1 = torch.atan2(start1[:, 0], start1[:, 1])
                start1 = start1.detach().squeeze().tolist()
                path1.append(start1)
                tree=1
                print(f'The start I is: {start1}')
            else:
                graph_const_bw = GraphConstructor(args.config_save_path, args.known_environment, start2, start1)
                graph_bw = graph_const_bw.construct_graph()

                eps = 1e-5
                if args.gaussian_normalization:
                    mean, std = load_normalization_info(args.gaussian_normalization)
                graph_bw['x'] = (graph_bw['x'] - mean) / (std + 1e-5)

                start2 = model(graph_bw)
                start2 = start2.reshape(6, 2)
                start2 = torch.atan2(start2[:, 0], start2[:, 1])
                start2 = start2.detach().squeeze().tolist()
                path2. append(start2)
                tree=0
                print(f'The start II is: {start2}')
            target_reached, _, _ = planning_module.steer(start1, start2, args.step_size, args.dof, col_times)
            print(f'Step number: {step}')
            print(f'Target is reached: {target_reached}')

        if (step > 100 or not target_reached):
            print("Planner was not able to plan.")

        if target_reached:
            for p1 in range(0, len(path1)):
                path.append(path1[p1])
            for p2 in range(len(path2)-1, -1, -1):
                path.append(path2[p2])

            path = planning_module.lazy_vertex_contraction(path, args.step_size, col_times)

            # Full collision checking
            feasible = planning_module.check_feasibility_entire_path(path,col_times)
            if feasible:
                t_end = time.time()
                planning_t = t_end - t_start 
                planning_times.append(planning_t)

                successful_paths = successful_paths + 1

                path_cost = planning_module.compute_cost(path, args.dof)
                planning_costs.append(path_cost)

                # Save path
                save_paths(args.planned_traj_path, path, args.known_environment, i)
                print(f'Planning got completed for path number: {i}')

            else:
                sp = 0
                feasible = False 
                step_size = args.step_size 

                while (not feasible) and (sp < 10) and (path !=0):
                    # Adaptive step size on replanning attempts.
                    if (sp==0):
                        step_size = 0.04
                    elif (sp == 2):
                        step_size = 0.03
                    elif (sp > 2):
                        step_size = 0.02
    planning_time = np.mean(np.array(planning_times))
    planning_time_std = np.std(np.array(planning_times))
    planning_cost = np.mean(np.array(planning_costs))
    planning_cost_std = np.std(np.array(planning_costs))
    unsuccessful_paths = args.num_sampled_start_end - successful_paths
    print(f'planning time for GNN, (Mean): {planning_time}, (std): {planning_time_std}')
    print(f'planning cost for GNN, (Mean): {planning_cost}, (std): {planning_cost_std}')
    print(f'Number of successful planned paths: {successful_paths}')
    save_planning_info(args.planned_traj_path, args.known_environment, planning_times, planning_costs, successful_paths, unsuccessful_paths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # GCN parameters
    parser.add_argument('--in_channels', type=int, default=3, help="Number of features for each joint")
    parser.add_argument('--out_channels', type=int, default=6, help="Number of goals (6 robot joints)")
    parser.add_argument('--add_self_loop', type=bool, default=True, help="Whether netwrok has self-loops")
    parser.add_argument('--normalize', type=bool, default=True, help="Normalize argument in FCNConv")
    parser.add_argument('--trained_model_path', type=str, default='/home/davood/catkin_ws/src/GNN2/src/trained_models', help='The path for saving trained models')
    parser.add_argument('--trained_model_name', type=str, default='gaussian_trained_GCN.pt', help='The name of the trained model.')
    parser.add_argument('--gaussian_normalization', type=bool, default=True, help='How the data has been normalized for training the GCN')


    # obstacle directories
    parser.add_argument('--obstacle_path', type=str, default='/home/davood/catkin_ws/src/GNN2/src/dataset', help='The directory of the obstacles')
    parser.add_argument('--num_workspaces', type=int, default=17, help='Number of workspaces')
    parser.add_argument('--WS_names', nargs='+', type=str, default=['WS_1', 'WS_2', 'WS_3', 'WS_4', 'WS_5', 'WS_6',
                                                                    'WS_7', 'WS_8', 'WS_9', 'WS_10', 'WS_11', 'WS_12',
                                                                    'WS_13', 'WS_14', 'WS_15', 'WS_16', 'WS_17'], help='Name of the workspaces')
    parser.add_argument('--known_environment', type=bool, default=True, help='Online planning in known environmnet')
    parser.add_argument('--num_start_end', type=int, default=200, help="The number of start - end configurations for each workspace.")
    parser.add_argument('--config_save_path', type=str, default='/home/davood/catkin_ws/src/GNN2/src/testing/start_goal_samples', help='The path to save the generated configs')

    # Online planning
    parser.add_argument('--num_sampled_start_end', type=int, default=200, help='Number of generated start-end configs. Just for testing purposes.')
    parser.add_argument('--dof', type=int, default=6, help='Number of degrees of freedom of the robot.')
    parser.add_argument('--step_size', type=float, default=0.05, help="Step size for steering purposes.")
    parser.add_argument('--forward_path', type=list, default=[], help='Forward planned path.')
    parser.add_argument('--backward_path', type=list, default=[], help='Backward planned path.')
    parser.add_argument('--planned_traj_path', type=str, default='/home/davood/catkin_ws/src/GNN2/src/testing/GCN_paper/planned_paths', help='The directory for saving planned paths.')

    args = parser.parse_args()
    main(args)
