#!/usr/bin/env python3

import sys
import rospy
import moveit_commander
import torch
import pickle
import os 

import moveit_msgs.msg
import geometry_msgs.msg 


class CheckFeasibilityGCN():
    def __init__(self):
        '''
        Checking feasiblity of the generated paths.
        '''
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('GCN_validation', anonymous=True)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        group_name = "ur5e_arm"
        self.move_group = moveit_commander.MoveGroupCommander(group_name)
        group_name = "gripper"
        self.hand_group = moveit_commander.MoveGroupCommander(group_name)

        self.display_trajectory_publisher = rospy.Publisher('move_group/display_planned_path',
                                                            moveit_msgs.msg.DisplayTrajectory,
                                                            queue_size=20)
        self.palanned_paths_directory = '/home/davood/catkin_ws/src/GNN2/src/testing/GCN_paper/planned_paths/unknown_environment'
        self.num_paths = 69
        self.load_environment()

    def load_environment(self):
        '''
        Loads the environment
        '''
        self.add_desks()
        self.add_obstacles_unknown()

    def load_paths(self, idx):
        '''
        Load saved paths
        '''
        with open(os.path.join(self.palanned_paths_directory, f'planned_path_{idx}.pkl'), 'rb') as file:
            path = pickle.load(file)

        return path

    def check_paths(self, path):
        '''
        Check the planned paths
        '''
        for waypoint in path:
            joint_goal = waypoint
            joint_goal = [float(point) for point in joint_goal]
            self.move_group.go(joint_goal, wait=True)

    def add_desks(self):
        '''
        Desks under the robot
        '''

        desk1_pose = geometry_msgs.msg.PoseStamped()
        desk1_pose.header.frame_id = self.robot.get_planning_frame()
        desk1_pose.pose.orientation.w = 1.0
        desk1_pose.pose.position.x = 0
        desk1_pose.pose.position.y = 0.6
        desk1_pose.pose.position.z = -0.075
        desk1_name = "desk_back"
        self.scene.add_box(desk1_name, desk1_pose, size=(2, 1.2, 0.1)) # Actual size of the desk in the lab!

        desk2_pose = geometry_msgs.msg.PoseStamped()
        desk2_pose.header.frame_id = self.robot.get_planning_frame()
        desk2_pose.pose.orientation.w = 1.0
        desk2_pose.pose.position.x = 0
        desk2_pose.pose.position.y = -0.6
        desk2_pose.pose.position.z = -0.075
        desk2_name = "desk"
        self.scene.add_box(desk2_name, desk2_pose, size=(2, 1.2, 0.1)) # Actual size of the desk in the lab!

    def add_obstacles_unknown(self):
        '''
        Obstacles for the unknown environment.
        1 - Same number of obstacles with the same size, but different locations.
        2 - Same number of obstacles with different size and different locations.
        3 - Different type of obstacles (different number, different shape, etc.)
        '''

        # Add Monitor
        monitor_pose = geometry_msgs.msg.PoseStamped()
        monitor_pose.header.frame_id = self.robot.get_planning_frame()
        monitor_pose.pose.orientation.w = 1.0
        monitor_pose.pose.position.x = 0.6
        monitor_pose.pose.position.y = 0.6
        monitor_pose.pose.position.z = 0.05
        monitor_name = "Monitor"
        self.scene.add_box(monitor_name, monitor_pose, size=[0.3, 0.17, 0.1])   # here even the size of the obstacles can be different - Not for now!

        # Add desktop
        desktop_pose = geometry_msgs.msg.PoseStamped()
        desktop_pose.header.frame_id = self.robot.get_planning_frame()
        desktop_pose.pose.orientation.w = 1.0
        desktop_pose.pose.position.x = 0.25
        desktop_pose.pose.position.y = -0.7
        desktop_pose.pose.position.z = 0.09
        desktop_name = "Desktop"
        self.scene.add_box(desktop_name, desktop_pose, size=[0.28, 0.28, 0.18])

        # Add Screw-driver
        screwdriver_pose = geometry_msgs.msg.PoseStamped()
        screwdriver_pose.header.frame_id = self.robot.get_planning_frame()
        screwdriver_pose.pose.orientation.w = 1.0
        screwdriver_pose.pose.position.x = 0.25
        screwdriver_pose.pose.position.y = 0.6
        screwdriver_pose.pose.position.z = 0.12
        screwdriver_name = "Screwdriver_box"
        self.scene.add_box(screwdriver_name, screwdriver_pose, size=[0.28, 0.17, 0.24])

        # Add Disassembly-container
        disassembly_pose = geometry_msgs.msg.PoseStamped()
        disassembly_pose.header.frame_id = self.robot.get_planning_frame()
        disassembly_pose.pose.orientation.w = 1.0
        disassembly_pose.pose.position.x = 0.4
        disassembly_pose.pose.position.y = 0
        disassembly_pose.pose.position.z = 0.05
        disassembly_name = "Disassembly_container"
        self.scene.add_box(disassembly_name, disassembly_pose, size=[0.3, 0.17, 0.1]) 


def main():
    '''
    Check the feasibility of planned paths
    '''
    feasibility_gcn = CheckFeasibilityGCN()
    idx = 3
    path = feasibility_gcn.load_paths(3)
    print(path)
    feasibility_gcn.check_paths(path)
    print(f'Path {idx} has been planned.')



if __name__ == "__main__":
    main()



