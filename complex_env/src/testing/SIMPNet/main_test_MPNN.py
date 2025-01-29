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
from planning_utils import trajectory_test, save_paths, load_model, load_normalization_info, save_planning_info, set_random_seed
from get_fk_offline import GetFK
from graph_constructor import GraphConstructor
from trained_gnn import MotionGenerator
from collision_checking import IsInCollision

# Import messages
import moveit_msgs.msg


def main(args):
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
    model = MotionGenerator(args.node_feature_size, args.obs_feature_size, args.output_size, args.batch_size, args.num_joints)
    model = load_model(model, args.trained_model_path, args.trained_model_name)

    # Online Planning
    planning_module = OnlinePlanning(args.config_save_path, args.known_environment, args.dof, args.step_size)
    planning_module.load_obstacles()
    planning_times = []
    planning_costs = []
    num_steps = 300
    successful_paths = 0
    set_random_seed(42)
    i = 0

    while i < args.num_sampled_start_end:
        start1, goal1 = planning_module.load_start_goal_config(i)
        goal2, start2 = planning_module.load_start_goal_config(i)

        fk = GetFK()
        start1_3d = fk.run_joint(start1)
        start2_3d = fk.run_joint(start2)
        goal1_3d = fk.run_joint(goal1)
        goal2_3d = fk.run_joint(goal2)

        path1 = []
        path1.append(start1)
        path2 = []
        path2.append(start2)
        path = []
        target_reached = False
        step = 0
        tree = 0
        t_start = time.time()
        #target_reached = planning_module.steer(start1, start2, args.step_size, args.dof)

        while (not target_reached) and step < num_steps:
            step = step + 1

            if tree == 0:
                graph_const_fw = GraphConstructor(args.config_save_path, args.known_environment, start1_3d, start2_3d, start1, start2, args.num_joints)
                graph_fw, obs_embeddings = graph_const_fw.construct_graph()
                
                eps = 1e-3
                if args.gaussian_normalization:
                    feature_mean, feature_std, obs_mean, obs_std = load_normalization_info(args.gaussian_normalization)
                graph_fw['x'] = (graph_fw['x'] - feature_mean) / (feature_std + eps)
                obs_embeddings = (obs_embeddings - obs_mean) / (obs_std + eps)
                model.train()
                start = model(graph_fw, obs_embeddings)
                start = start.reshape(6, 2)
                start = torch.atan2(start[:, 0], start[:, 1])
                start = start.detach().squeeze().tolist()
                start1 = start
                start1_3d = fk.run_joint(start1)
                path1.append(start1)
                tree=1

            else:
                graph_const_bw = GraphConstructor(args.config_save_path, args.known_environment, start2_3d, start1_3d, start2, start1, args.num_joints)
                graph_bw, obs_embeddings = graph_const_bw.construct_graph()
                
                eps = 1e-3
                if args.gaussian_normalization:
                    feature_mean, feature_std, obs_mean, obs_std = load_normalization_info(args.gaussian_normalization)
                graph_bw['x'] = (graph_bw['x'] - feature_mean) / (feature_std + eps)
                obs_embeddings = (obs_embeddings - obs_mean) / (obs_std + eps)
                model.train()
                start = model(graph_bw, obs_embeddings)
                start = start.reshape(6, 2)
                start = torch.atan2(start[:, 0], start[:, 1])
                start = start.detach().squeeze().tolist()
                start2 = start
                start2_3d = fk.run_joint(start2)
                path2.append(start2)
                tree = 0

            target_reached= planning_module.steer(start1, start2, args.step_size, args.dof)
        
        if (step > num_steps or not target_reached):
            print("Planner was not able to plan.")

        if target_reached:
            for p1 in range(0, len(path1)):
                path.append(path1[p1])
            for p2 in range(len(path2)-1, -1, -1):
                path.append(path2[p2])

            path = planning_module.lazy_vertex_contraction(path, args.step_size)

            # Full collision checking
            feasible = planning_module.check_feasibility_entire_path(path)
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

                while (not feasible) and (sp < 5) and (path !=0):
                    # Adaptive step size on replanning attempts
                    print(f'replanning try: {sp}')
                    if (sp==0):
                        step_size = 0.04
                    elif (sp == 2):
                        step_size = 0.03
                    elif (sp > 2):
                        step_size = 0.02

                    sp = sp + 1
                    g = goal1
                    
                    # Neural replanning
                    step = 0
                    path_free = []
                    path_free.append(path[0])
                    for j in range(1, len(path) - 1):
                        collision = IsInCollision(path[j])
                        if not collision:
                            path_free.append(path[j])
                    path_free.append(g)
                    
                    new_path = []
                    for j in range(0, len(path_free) - 1):
                        target_reached = False

                        st = path_free[j]
                        gl = path_free[j+1]
                        steer= planning_module.steer(st, gl, step_size, args.dof)
                        if steer:
                            new_path.append(st)
                            new_path.append(gl)
                        else:
                            itr = 0
                            pA = []
                            pA.append(st)
                            pB = []
                            pB.append(gl)
                            while (not target_reached) and itr < num_steps:
                                itr = itr + 1
                                if tree == 0:
                                    st_3D = fk.run_joint(st)
                                    gl_3D = fk.run_joint(gl)
                                    graph_const_fw = GraphConstructor(args.config_save_path, args.known_environment, st_3D, gl_3D, st, gl, args.num_joints)
                                    graph_fw, obs_embeddings = graph_const_fw.construct_graph()
                                    
                                    eps = 1e-3
                                    if args.gaussian_normalization:
                                        feature_mean, _feature_std, obs_mean, obs_std = load_normalization_info(args.gaussian_normalization)
                                    graph_fw['x'] = (graph_fw['x'] - feature_mean) / (_feature_std + eps)
                                    obs_embeddings = (obs_embeddings - obs_mean) / (obs_std + eps)
                                    
                                    model.train()
                                    st = model(graph_fw, obs_embeddings)
                                    st = st.reshape(6, 2)
                                    st = torch.atan2(st[:, 0], st[:, 1])
                                    st = st.detach().squeeze().tolist()
                                    pA.append(st)
                                    tree=1
                                else:
                                    st_3D = fk.run_joint(st)
                                    gl_3D = fk.run_joint(gl)
                                    graph_const_bw = GraphConstructor(args.config_save_path, args.known_environment, gl_3D, st_3D, gl, st, args.num_joints)
                                    graph_bw, obs_embeddings = graph_const_bw.construct_graph()
                                    
                                    eps = 1e-3
                                    if args.gaussian_normalization:
                                        feature_mean, _feature_std, obs_mean, obs_std = load_normalization_info(args.gaussian_normalization)
                                    graph_bw['x'] = (graph_bw['x'] - feature_mean) / (_feature_std + eps)
                                    obs_embeddings = (obs_embeddings - obs_mean) / (obs_std + eps)

                                    model.train()
                                    gl = model(graph_bw, obs_embeddings)
                                    gl = gl.reshape(6, 2)
                                    gl = torch.atan2(gl[:, 0], gl[:, 1])
                                    gl = gl.detach().squeeze().tolist()
                                    pB.append(gl)
                                    tree=0
                                target_reached = planning_module.steer(st, gl, step_size, args.dof)
                            if not target_reached:
                                print('Failed to replan!')
                            else:
                                for p1 in range(0, len(pA)):
                                    new_path.append(pA[p1])
                                for p2 in range(len(pB)-1, -1, -1):
                                    new_path.append(pB[p2])
                    if new_path != 0:
                        new_path = planning_module.lazy_vertex_contraction(new_path, step_size)

                        # Full collision check
                        feasible = planning_module.check_feasibility_entire_path(new_path)
                        if feasible:
                            t_end = time.time()
                            planning_t = t_end - t_start
                            planning_times.append(planning_t)

                            successful_paths = successful_paths + 1

                            path_cost = planning_module.compute_cost(new_path, args.dof)
                            planning_costs.append(path_cost)

                            # Save path
                            save_paths(args.planned_traj_path, new_path, args.known_environment, i)
                            print(f'Planning got completed for path number: {i}')
                        else:
                            path = new_path
        i = i + 1
    
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
    parser.add_argument('--node_feature_size', type=int, default=16, help="Each node feature size.")
    parser.add_argument('--obs_feature_size', type=int, default=42, help="Size of obstacle embeddings for each environment.")
    parser.add_argument('--output_size', type=int, default=2, help="The size of output for each joint.")
    parser.add_argument('--batch_size', type=int, default=256, help="The batch size for evaluation.")
    parser.add_argument('--num_joints', type=int, default=6, help='Number of robot joints')
    parser.add_argument('--trained_model_path', type=str, default='/home/davood/catkin_ws/src/gnn4/src/trained_models', help='The path for saving trained models')
    parser.add_argument('--trained_model_name', type=str, default='gaussian_trained_mainMPNN.pt', help='The name of the trained model.')
    parser.add_argument('--gaussian_normalization', type=bool, default=True, help='How the data has been normalized for training the GCN')


    # obstacle directories
    parser.add_argument('--obstacle_path', type=str, default='/home/davood/catkin_ws/src/gnn4/src/dataset', help='The directory of the obstacles')
    parser.add_argument('--num_workspaces', type=int, default=11, help='Number of workspaces')
    parser.add_argument('--WS_names', nargs='+', type=str, default=['WS_1', 'WS_2', 'WS_3', 'WS_4', 'WS_5', 'WS_6',
                                                                    'WS_7', 'WS_8', 'WS_9', 'WS_10', 'WS_11'], help='Name of the workspaces')
    parser.add_argument('--known_environment', type=bool, default=False, help='Online planning in known environmnet')
    parser.add_argument('--num_start_end', type=int, default=200, help="The number of start - end configurations for each workspace.")
    parser.add_argument('--config_save_path', type=str, default='/home/davood/catkin_ws/src/gnn4/src/testing/start_goal_samples', help='The path to save the generated configs')

    # Online planning
    parser.add_argument('--num_sampled_start_end', type=int, default=200, help='Number of generated start-end configs. Just for testing purposes.')
    parser.add_argument('--dof', type=int, default=6, help='Number of degrees of freedom of the robot.')
    parser.add_argument('--step_size', type=float, default=0.05, help="Step size for steering purposes.")
    parser.add_argument('--forward_path', type=list, default=[], help='Forward planned path.')
    parser.add_argument('--backward_path', type=list, default=[], help='Backward planned path.')
    parser.add_argument('--planned_traj_path', type=str, default='/home/davood/catkin_ws/src/gnn4/src/testing/main_MPNN/planned_paths', help='The directory for saving planned paths.')

    args = parser.parse_args()
    main(args)
