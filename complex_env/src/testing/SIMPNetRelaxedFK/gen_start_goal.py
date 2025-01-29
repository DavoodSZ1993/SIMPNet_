#!/usr/bin/env python3

import numpy as np
import os 
import pickle

# Custom Modules
from collision_checking import IsInCollision

# Messages and services
import geometry_msgs.msg
from moveit_msgs.msg import RobotTrajectory, RobotState
from sensor_msgs.msg import JointState

class StartGoalConfig:
    X_LIM = [0.1, 1.0]
    Y_LIM = [-1.0, 1.0]
    Z_LIM = [0.1, 1.0]
    JOINT_NAMES = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    def __init__(self, move_group, num_start_end: int, path: str, known_environment: bool)->None:
        '''
        Here the limits are defined in workspace, becasue we want the robot work within the workspace.
        '''
        self.num_start_end = num_start_end
        self.move_group = move_group
        self.path = path
        self.known_environment = known_environment

        self.config_generation()


    def config_generation(self):

        self.move_group.set_planner_id("RRTstar")
        self.move_group.set_planning_time(5)

        num_configs = 0

        init_state = self.move_group.get_current_joint_values()
        joint_state = JointState()
        joint_state.name = StartGoalConfig.JOINT_NAMES
        joint_state.position = init_state

        robot_state = RobotState()
        robot_state.joint_state = joint_state

        while num_configs < self.num_start_end:
            self.move_group.set_start_state(robot_state)
            x = np.random.uniform(StartGoalConfig.X_LIM[0], StartGoalConfig.X_LIM[1])
            y = np.random.uniform(StartGoalConfig.Y_LIM[0], StartGoalConfig.Y_LIM[1])
            z = np.random.uniform(StartGoalConfig.Z_LIM[0], StartGoalConfig.Z_LIM[1])

            goal_pose = geometry_msgs.msg.Pose()
            goal_pose.position.x = x
            goal_pose.position.y = y
            goal_pose.position.z = z

            self.move_group.set_pose_target(goal_pose)
            plan = self.move_group.plan(goal_pose)
            success = self.move_group.go(wait=True)

            # Save config
            trajectory = RobotTrajectory()
            (_, trajectory, _, _) = plan
            pos = [point.positions for point in trajectory.joint_trajectory.points]
            if pos != []:
                in_collision = IsInCollision(pos[-1])
                if not in_collision:
                    # Save start - goal configuration 
                    if self.known_environment:
                        env_path  = os.path.join(self.path, 'known_environment')
                        if not os.path.exists(env_path):
                            os.makedirs(env_path)

                        start_config = pos[0]
                        end_config = pos[-1]
                        config = {'start config': start_config,
                                  'end config': end_config}
                        name = f'config_{num_configs}.pkl'
                        with open(os.path.join(env_path, name), 'wb') as file:
                            pickle.dump(config, file)
                    else:
                        env_path  = os.path.join(self.path, 'unknown_environment')
                        if not os.path.exists(env_path):
                            os.makedirs(env_path)

                        start_config = pos[0]
                        end_config = pos[-1]
                        config = {'start config': start_config,
                                  'end config': end_config}
                        name = f'config_{num_configs}.pkl'
                        with open(os.path.join(env_path, name), 'wb') as file:
                            pickle.dump(config, file)
                    joint_state.position = pos[-1]
                    robot_state.joint_state = joint_state
                    print(f'Start-Goal configuration generated successfully!: {num_configs}')
                    num_configs = num_configs + 1







