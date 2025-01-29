#!/usr/bin/env python3

import rospy
import moveit_commander
import geometry_msgs.msg 

import math 
import os
import numpy as np
import pickle 


class LoadEnvironment:
    def __init__(self, robot, scene, hand_group, obstacle_path, num_workspaces, WS_names, known_environment, config_save_path):
        
        self.robot = robot 
        self.scene = scene
        self.hand_group = hand_group
        self.obstacle_path = obstacle_path
        self.num_workspaces = num_workspaces
        self.ws_names = WS_names
        self.config_save_path = config_save_path
        self.unknown_env_obs_embed = {}

        self.ws_path = os.path.join(self.obstacle_path, self.ws_names[0])
        obstacle_path = os.path.join(self.ws_path, 'obs_embeddings.pkl')
        with open(obstacle_path, 'rb') as file:
            self.obs_embeddings = pickle.load(file)
            print(f'Obstacle embedding for environment {self.ws_names[0]}: {self.obs_embeddings}')

        self.add_desks()
        if known_environment:
            self.add_obstacles_known()
            env_path = os.path.join(self.config_save_path, 'known_environment')
            if not os.path.exists(env_path):
                os.makedirs(env_path)

            with open(os.path.join(env_path, 'obs_embeddings.pkl'), 'wb') as file:
                pickle.dump(self.obs_embeddings, file)
        else:
            self.add_obstacles_unknown()
            #self.add_obstacles_unknown1()

        self.close_gripper()

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

        desk3_pose = geometry_msgs.msg.PoseStamped()
        desk3_pose.header.frame_id = self.robot.get_planning_frame()
        desk3_pose.pose.orientation.w = 1.0
        desk3_pose.pose.position.x = -0.4
        desk3_pose.pose.position.y = 0.0
        desk3_pose.pose.position.z = 0.4
        desk3_name = "desk_behind"
        self.scene.add_box(desk3_name, desk3_pose, size=(0.5, 2.4, 1)) # 

    def add_obstacles_known(self):

        # Add Monitor
        monitor_pose = geometry_msgs.msg.PoseStamped()
        monitor_pose.header.frame_id = self.robot.get_planning_frame()
        monitor_pose.pose.orientation.w = 1.0
        monitor_pose.pose.position.x = self.obs_embeddings['Monitor']['center'][0]
        monitor_pose.pose.position.y = self.obs_embeddings['Monitor']['center'][1]
        monitor_pose.pose.position.z = self.obs_embeddings['Monitor']['center'][2]
        monitor_name = "Monitor"
        self.scene.add_box(monitor_name, monitor_pose, size=self.obs_embeddings['Monitor']['size'])

        # Add desktop
        desktop_pose = geometry_msgs.msg.PoseStamped()
        desktop_pose.header.frame_id = self.robot.get_planning_frame()
        desktop_pose.pose.orientation.w = 1.0
        desktop_pose.pose.position.x = self.obs_embeddings['Desktop']['center'][0]
        desktop_pose.pose.position.y = self.obs_embeddings['Desktop']['center'][1]
        desktop_pose.pose.position.z = self.obs_embeddings['Desktop']['center'][2]
        desktop_name = "Desktop"
        self.scene.add_box(desktop_name, desktop_pose, size=self.obs_embeddings['Desktop']['size'])

        # Add Screw-driver
        screwdriver_pose = geometry_msgs.msg.PoseStamped()
        screwdriver_pose.header.frame_id = self.robot.get_planning_frame()
        screwdriver_pose.pose.orientation.w = 1.0
        screwdriver_pose.pose.position.x = self.obs_embeddings['Screwdriver_box']['center'][0]
        screwdriver_pose.pose.position.y = self.obs_embeddings['Screwdriver_box']['center'][1]
        screwdriver_pose.pose.position.z = self.obs_embeddings['Screwdriver_box']['center'][2]
        screwdriver_name = "Screwdriver_box"
        self.scene.add_box(screwdriver_name, screwdriver_pose, size=self.obs_embeddings['Screwdriver_box']['size'])

        # Add Disassembly-container
        disassembly_pose = geometry_msgs.msg.PoseStamped()
        disassembly_pose.header.frame_id = self.robot.get_planning_frame()
        disassembly_pose.pose.orientation.w = 1.0
        disassembly_pose.pose.position.x = self.obs_embeddings['Disassembly_container']['center'][0]
        disassembly_pose.pose.position.y = self.obs_embeddings['Disassembly_container']['center'][1]
        disassembly_pose.pose.position.z = self.obs_embeddings['Disassembly_container']['center'][2]
        disassembly_name = "Screwdriver_box"
        self.scene.add_box(disassembly_name, disassembly_pose, size=self.obs_embeddings['Disassembly_container']['size'])

        # Add Desktop I
        desktop1_pose = geometry_msgs.msg.PoseStamped()
        desktop1_pose.header.frame_id = self.robot.get_planning_frame()
        desktop1_pose.pose.orientation.w = 1.0
        desktop1_pose.pose.position.x = self.obs_embeddings['Desktop I']['center'][0]
        desktop1_pose.pose.position.y = self.obs_embeddings['Desktop I']['center'][1]
        desktop1_pose.pose.position.z = self.obs_embeddings['Desktop I']['center'][2]
        desktop1_name = "Desktop I"
        self.scene.add_box(desktop1_name, desktop1_pose, size=self.obs_embeddings['Desktop I']['size'])

        # Add Desktop II
        desktop2_pose = geometry_msgs.msg.PoseStamped()
        desktop2_pose.header.frame_id = self.robot.get_planning_frame()
        desktop2_pose.pose.orientation.w = 1.0
        desktop2_pose.pose.position.x = self.obs_embeddings['Desktop II']['center'][0]
        desktop2_pose.pose.position.y = self.obs_embeddings['Desktop II']['center'][1]
        desktop2_pose.pose.position.z = self.obs_embeddings['Desktop II']['center'][2]
        desktop2_name = "Desktop II"
        self.scene.add_box(desktop2_name, desktop2_pose, size=self.obs_embeddings['Desktop II']['size'])

        # Add Disassembly-container
        disassembly1_pose = geometry_msgs.msg.PoseStamped()
        disassembly1_pose.header.frame_id = self.robot.get_planning_frame()
        disassembly1_pose.pose.orientation.w = 1.0
        disassembly1_pose.pose.position.x = self.obs_embeddings['Disassembly_container_II']['center'][0]
        disassembly1_pose.pose.position.y = self.obs_embeddings['Disassembly_container_II']['center'][1]
        disassembly1_pose.pose.position.z = self.obs_embeddings['Disassembly_container_II']['center'][2]
        disassembly1_name = "Disassembly_container II"
        self.scene.add_box(disassembly1_name, disassembly1_pose, size=self.obs_embeddings['Disassembly_container_II']['size'])       


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
        self.scene.add_box(monitor_name, monitor_pose, size=self.obs_embeddings['Monitor']['size'])   # here even the size of the obstacles can be different - Not for now!

        center_x = monitor_pose.pose.position.x
        center_y = monitor_pose.pose.position.y
        center_z = monitor_pose.pose.position.z
        size_x = self.obs_embeddings['Monitor']['size'][0]
        size_y = self.obs_embeddings['Monitor']['size'][1]
        size_z = self.obs_embeddings['Monitor']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Monitor']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Monitor'] = obs_info


        # Add desktop
        desktop_pose = geometry_msgs.msg.PoseStamped()
        desktop_pose.header.frame_id = self.robot.get_planning_frame()
        desktop_pose.pose.orientation.w = 1.0
        desktop_pose.pose.position.x = 0.25
        desktop_pose.pose.position.y = -0.7
        desktop_pose.pose.position.z = 0.09
        desktop_name = "Desktop"
        self.scene.add_box(desktop_name, desktop_pose, size=self.obs_embeddings['Desktop']['size'])

        center_x = desktop_pose.pose.position.x
        center_y = desktop_pose.pose.position.y
        center_z = desktop_pose.pose.position.z
        size_x = self.obs_embeddings['Desktop']['size'][0]
        size_y = self.obs_embeddings['Desktop']['size'][1]
        size_z = self.obs_embeddings['Desktop']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Desktop']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Desktop'] = obs_info
        

        # Add Screw-driver
        screwdriver_pose = geometry_msgs.msg.PoseStamped()
        screwdriver_pose.header.frame_id = self.robot.get_planning_frame()
        screwdriver_pose.pose.orientation.w = 1.0
        screwdriver_pose.pose.position.x = 0.25
        screwdriver_pose.pose.position.y = 0.6
        screwdriver_pose.pose.position.z = 0.12
        screwdriver_name = "Screwdriver_box"
        self.scene.add_box(screwdriver_name, screwdriver_pose, size=self.obs_embeddings['Screwdriver_box']['size'])

        center_x = screwdriver_pose.pose.position.x
        center_y = screwdriver_pose.pose.position.y
        center_z = screwdriver_pose.pose.position.z
        size_x = self.obs_embeddings['Screwdriver_box']['size'][0]
        size_y = self.obs_embeddings['Screwdriver_box']['size'][1]
        size_z = self.obs_embeddings['Screwdriver_box']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Screwdriver_box']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Screwdriver_box'] = obs_info

        # Add Disassembly-container
        disassembly_pose = geometry_msgs.msg.PoseStamped()
        disassembly_pose.header.frame_id = self.robot.get_planning_frame()
        disassembly_pose.pose.orientation.w = 1.0
        disassembly_pose.pose.position.x = 0.4
        disassembly_pose.pose.position.y = 0
        disassembly_pose.pose.position.z = 0.05
        disassembly_name = "Disassembly_container"
        self.scene.add_box(disassembly_name, disassembly_pose, size=self.obs_embeddings['Disassembly_container']['size']) 

        center_x = disassembly_pose.pose.position.x
        center_y = disassembly_pose.pose.position.y
        center_z = disassembly_pose.pose.position.z
        size_x = self.obs_embeddings['Disassembly_container']['size'][0]
        size_y = self.obs_embeddings['Disassembly_container']['size'][1]
        size_z = self.obs_embeddings['Disassembly_container']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Disassembly_container']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Disassembly_container'] = obs_info

        # Add Desktop I
        desktop1_pose = geometry_msgs.msg.PoseStamped()
        desktop1_pose.header.frame_id = self.robot.get_planning_frame()
        desktop1_pose.pose.orientation.w = 1.0
        desktop1_pose.pose.position.x = self.obs_embeddings['Desktop I']['center'][0]
        desktop1_pose.pose.position.y = self.obs_embeddings['Desktop I']['center'][1]
        desktop1_pose.pose.position.z = self.obs_embeddings['Desktop I']['center'][2]
        desktop1_name = "Desktop I"
        self.scene.add_box(desktop1_name, desktop1_pose, size=self.obs_embeddings['Desktop I']['size'])

        center_x = desktop1_pose.pose.position.x
        center_y = desktop1_pose.pose.position.y
        center_z = desktop1_pose.pose.position.z
        size_x = self.obs_embeddings['Desktop I']['size'][0]
        size_y = self.obs_embeddings['Desktop I']['size'][1]
        size_z = self.obs_embeddings['Desktop I']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Desktop I']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Desktop I'] = obs_info

        # Add Desktop II
        desktop2_pose = geometry_msgs.msg.PoseStamped()
        desktop2_pose.header.frame_id = self.robot.get_planning_frame()
        desktop2_pose.pose.orientation.w = 1.0
        desktop2_pose.pose.position.x = self.obs_embeddings['Desktop II']['center'][0]
        desktop2_pose.pose.position.y = self.obs_embeddings['Desktop II']['center'][1]
        desktop2_pose.pose.position.z = self.obs_embeddings['Desktop II']['center'][2]
        desktop2_name = "Desktop II"
        self.scene.add_box(desktop2_name, desktop2_pose, size=self.obs_embeddings['Desktop II']['size'])

        center_x = desktop2_pose.pose.position.x
        center_y = desktop2_pose.pose.position.y
        center_z = desktop2_pose.pose.position.z
        size_x = self.obs_embeddings['Desktop II']['size'][0]
        size_y = self.obs_embeddings['Desktop II']['size'][1]
        size_z = self.obs_embeddings['Desktop II']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Desktop II']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Desktop II'] = obs_info

        # Add Disassembly-container
        disassembly1_pose = geometry_msgs.msg.PoseStamped()
        disassembly1_pose.header.frame_id = self.robot.get_planning_frame()
        disassembly1_pose.pose.orientation.w = 1.0
        disassembly1_pose.pose.position.x = self.obs_embeddings['Disassembly_container_II']['center'][0]
        disassembly1_pose.pose.position.y = self.obs_embeddings['Disassembly_container_II']['center'][1]
        disassembly1_pose.pose.position.z = self.obs_embeddings['Disassembly_container_II']['center'][2]
        disassembly1_name = "Disassembly_container II"
        self.scene.add_box(disassembly1_name, disassembly1_pose, size=self.obs_embeddings['Disassembly_container_II']['size'])

        center_x = disassembly1_pose.pose.position.x
        center_y = disassembly1_pose.pose.position.y
        center_z = disassembly1_pose.pose.position.z
        size_x = self.obs_embeddings['Disassembly_container_II']['size'][0]
        size_y = self.obs_embeddings['Disassembly_container_II']['size'][1]
        size_z = self.obs_embeddings['Disassembly_container_II']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Disassembly_container_II']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Disassembly_container_II'] = obs_info

        # Save obstacle embeddings
        env_path = os.path.join(self.config_save_path, 'unknown_environment')
        if not os.path.exists(env_path):
            os.makedirs(env_path)

        with open(os.path.join(env_path, 'obs_embeddings.pkl'), 'wb') as file:
            pickle.dump(self.unknown_env_obs_embed, file)
            print(f'Obstacle embeddings of the unknown environment: {self.unknown_env_obs_embed}')

    def add_obstacles_unknown1(self):
        '''
        Obstacles for the second unknown environment.
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
        self.scene.add_box(monitor_name, monitor_pose, size=self.obs_embeddings['Monitor']['size'])   # here even the size of the obstacles can be different - Not for now!

        center_x = monitor_pose.pose.position.x
        center_y = monitor_pose.pose.position.y
        center_z = monitor_pose.pose.position.z
        size_x = self.obs_embeddings['Monitor']['size'][0]
        size_y = self.obs_embeddings['Monitor']['size'][1]
        size_z = self.obs_embeddings['Monitor']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Monitor']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Monitor'] = obs_info


        # Add desktop
        desktop_pose = geometry_msgs.msg.PoseStamped()
        desktop_pose.header.frame_id = self.robot.get_planning_frame()
        desktop_pose.pose.orientation.w = 1.0
        desktop_pose.pose.position.x = 0.5
        desktop_pose.pose.position.y = 0.0
        desktop_pose.pose.position.z = 0.09
        desktop_name = "Desktop"
        self.scene.add_box(desktop_name, desktop_pose, size=self.obs_embeddings['Desktop']['size'])

        center_x = desktop_pose.pose.position.x
        center_y = desktop_pose.pose.position.y
        center_z = desktop_pose.pose.position.z
        size_x = self.obs_embeddings['Desktop']['size'][0]
        size_y = self.obs_embeddings['Desktop']['size'][1]
        size_z = self.obs_embeddings['Desktop']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Desktop']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Desktop'] = obs_info

        # Add Screw-driver
        screwdriver_pose = geometry_msgs.msg.PoseStamped()
        screwdriver_pose.header.frame_id = self.robot.get_planning_frame()
        screwdriver_pose.pose.orientation.w = 1.0
        screwdriver_pose.pose.position.x = 0.25
        screwdriver_pose.pose.position.y = 0.6
        screwdriver_pose.pose.position.z = 0.12
        screwdriver_name = "Screwdriver_box"
        self.scene.add_box(screwdriver_name, screwdriver_pose, size=self.obs_embeddings['Screwdriver_box']['size'])

        center_x = screwdriver_pose.pose.position.x
        center_y = screwdriver_pose.pose.position.y
        center_z = screwdriver_pose.pose.position.z
        size_x = self.obs_embeddings['Screwdriver_box']['size'][0]
        size_y = self.obs_embeddings['Screwdriver_box']['size'][1]
        size_z = self.obs_embeddings['Screwdriver_box']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Screwdriver_box']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Screwdriver_box'] = obs_info

        # Add Disassembly-container
        disassembly_pose = geometry_msgs.msg.PoseStamped()
        disassembly_pose.header.frame_id = self.robot.get_planning_frame()
        disassembly_pose.pose.orientation.w = 1.0
        disassembly_pose.pose.position.x = 0.4
        disassembly_pose.pose.position.y = -0.7
        disassembly_pose.pose.position.z = 0.05
        disassembly_name = "Disassembly_container"
        self.scene.add_box(disassembly_name, disassembly_pose, size=self.obs_embeddings['Disassembly_container']['size']) 

        center_x = disassembly_pose.pose.position.x
        center_y = disassembly_pose.pose.position.y
        center_z = disassembly_pose.pose.position.z
        size_x = self.obs_embeddings['Disassembly_container']['size'][0]
        size_y = self.obs_embeddings['Disassembly_container']['size'][1]
        size_z = self.obs_embeddings['Disassembly_container']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Disassembly_container']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Disassembly_container'] = obs_info

        # Add Desktop I
        desktop1_pose = geometry_msgs.msg.PoseStamped()
        desktop1_pose.header.frame_id = self.robot.get_planning_frame()
        desktop1_pose.pose.orientation.w = 1.0
        desktop1_pose.pose.position.x = self.obs_embeddings['Desktop I']['center'][0]
        desktop1_pose.pose.position.y = self.obs_embeddings['Desktop I']['center'][1]
        desktop1_pose.pose.position.z = self.obs_embeddings['Desktop I']['center'][2]
        desktop1_name = "Desktop I"
        self.scene.add_box(desktop1_name, desktop1_pose, size=self.obs_embeddings['Desktop I']['size'])

        center_x = desktop1_pose.pose.position.x
        center_y = desktop1_pose.pose.position.y
        center_z = desktop1_pose.pose.position.z
        size_x = self.obs_embeddings['Desktop I']['size'][0]
        size_y = self.obs_embeddings['Desktop I']['size'][1]
        size_z = self.obs_embeddings['Desktop I']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Desktop I']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Desktop I'] = obs_info

        # Add Desktop II
        desktop2_pose = geometry_msgs.msg.PoseStamped()
        desktop2_pose.header.frame_id = self.robot.get_planning_frame()
        desktop2_pose.pose.orientation.w = 1.0
        desktop2_pose.pose.position.x = self.obs_embeddings['Desktop II']['center'][0]
        desktop2_pose.pose.position.y = self.obs_embeddings['Desktop II']['center'][1]
        desktop2_pose.pose.position.z = self.obs_embeddings['Desktop II']['center'][2]
        desktop2_name = "Desktop II"
        self.scene.add_box(desktop2_name, desktop2_pose, size=self.obs_embeddings['Desktop II']['size'])

        center_x = desktop2_pose.pose.position.x
        center_y = desktop2_pose.pose.position.y
        center_z = desktop2_pose.pose.position.z
        size_x = self.obs_embeddings['Desktop II']['size'][0]
        size_y = self.obs_embeddings['Desktop II']['size'][1]
        size_z = self.obs_embeddings['Desktop II']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Desktop II']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Desktop II'] = obs_info

        # Add Disassembly-container
        disassembly1_pose = geometry_msgs.msg.PoseStamped()
        disassembly1_pose.header.frame_id = self.robot.get_planning_frame()
        disassembly1_pose.pose.orientation.w = 1.0
        disassembly1_pose.pose.position.x = self.obs_embeddings['Disassembly_container_II']['center'][0]
        disassembly1_pose.pose.position.y = self.obs_embeddings['Disassembly_container_II']['center'][1]
        disassembly1_pose.pose.position.z = self.obs_embeddings['Disassembly_container_II']['center'][2]
        disassembly1_name = "Disassembly_container II"
        self.scene.add_box(disassembly1_name, disassembly1_pose, size=self.obs_embeddings['Disassembly_container_II']['size'])

        center_x = disassembly1_pose.pose.position.x
        center_y = disassembly1_pose.pose.position.y
        center_z = disassembly1_pose.pose.position.z
        size_x = self.obs_embeddings['Disassembly_container_II']['size'][0]
        size_y = self.obs_embeddings['Disassembly_container_II']['size'][1]
        size_z = self.obs_embeddings['Disassembly_container_II']['size'][2]

        obs_edges = np.zeros((4, 3))
        obs_edges[0, :] = [center_x - size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[1, :] = [center_x - size_x/2, center_y + size_y/2, center_z + size_z/2]
        obs_edges[2, :] = [center_x + size_x/2, center_y - size_y/2, center_z + size_z/2]
        obs_edges[3, :] = [center_x + size_x/2, center_y + size_y/2, center_z + size_z/2]

        obs_info = {}
        obs_info['size'] = self.obs_embeddings['Disassembly_container_II']['size']
        obs_info['center'] = [center_x, center_y, center_z]
        obs_info['edges'] = obs_edges
        self.unknown_env_obs_embed['Disassembly_container_II'] = obs_info

        # Save obstacle embeddings
        env_path = os.path.join(self.config_save_path, 'unknown_environment1')
        if not os.path.exists(env_path):
            os.makedirs(env_path)

        with open(os.path.join(env_path, 'obs_embeddings.pkl'), 'wb') as file:
            pickle.dump(self.unknown_env_obs_embed, file)
            print(f'Obstacle embeddings of the unknown environment: {self.unknown_env_obs_embed}')


    def close_gripper(self):
        '''
        Close the gripper for planning.
        '''

        self.hand_group.set_named_target("closed")
        success = self.hand_group.go(wait=True)
        self.hand_group.stop()