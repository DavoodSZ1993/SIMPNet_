#!/usr/bin/env python3

import rospy 
import numpy as np 
import os 

from moveit_msgs.srv import GetPositionFK
from moveit_msgs.srv import GetPositionFKRequest
from moveit_msgs.srv import GetPositionFKResponse
from sensor_msgs.msg import JointState

from tf.transformations import euler_from_quaternion

# Class to make FK calls using the /compute_fk service

class GetFK(object):
    def __init__(self, traj:list)->None:
        '''
        A class to do FK calls through the MoveIt /compute_fk service
        :param traj: the trajectory that the Forward Kinematics needs to be operate on.
        '''

        rospy.init_node('get_joint_position', anonymous=True)
        self.rate = rospy.Rate(0.1)

        rospy.loginfo('Initializing GetFK...')
        self.fk_link = ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']
        self.frame_id = 'base_link'
        rospy.loginfo(f'PoseStamped answers will be on frame: {self.frame_id}')

        self.fk_srv = rospy.ServiceProxy('/compute_fk', GetPositionFK)
        rospy.loginfo('Waiting for /compute_fk service....')
        self.fk_srv.wait_for_service()
        rospy.loginfo('Connected!')

        self.joint_states = []
        self.link_poses = {}
        self.link_orientation = {}
        self.joint_poses = []
        self.directory = '/home/davood/catkin_ws/src/GNN2/src/dataset_FK'
        self.trajectory = traj 
    
    def get_joint_states(self)->None:
        '''
        Gets each joint in the trajectory.
        '''
        
        for i in range(len(self.trajectory)):
            # position = self.trajectory[i]
            position = self.wrap_angle(list(self.trajectory[i]))      # This is for making all the angles between -pi - pi (dataset FK1)
            joint_state = JointState()
            joint_state.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            joint_state.position = position
            self.joint_states.append(joint_state)

    def get_current_fk(self, joint_idx:int)->None:
        '''
        Forward Kinematics
        '''
        while not rospy.is_shutdown() and self.joint_states[joint_idx] is None:
            rospy.logwarn('Waiting for a /joint_states message...')
            self.rate.sleep()
        return self.get_fk(joint_idx)

    def get_fk(self, joint_idx:int):
        '''
        Do an FK call
        '''
        if self.fk_link is None:
            fk_link = self.fk_link

        req = GetPositionFKRequest()
        req.header.frame_id = self.frame_id
        req.fk_link_names = self.fk_link
        req.robot_state.joint_state = self.joint_states[joint_idx]

        try:
            resp = self.fk_srv.call(req)
            return resp
        except rospy.ServiceException as e:
            rospy.logerr(f'Service exception: {str(e)}')
            resp = GetPositionFKResponse()
            resp.error_code = 99999                      # Failure
            return resp

    def run(self)->list:
        '''
        Forward Kinematics Serice
        '''
        self.get_joint_states()

        for i in range(len(self.joint_states)):
            resp = self.get_current_fk(i)
            positions = []
            for j in range(len(resp.pose_stamped)):
                x = resp.pose_stamped[j].pose.position.x
                y = resp.pose_stamped[j].pose.position.y
                z = resp.pose_stamped[j].pose.position.z

                position = [x, y, z]
                positions.append(position)
                self.link_poses[resp.fk_link_names[j]] = position 

                qx = resp.pose_stamped[j].pose.orientation.x
                qy = resp.pose_stamped[j].pose.orientation.y
                qz = resp.pose_stamped[j].pose.orientation.z
                qw = resp.pose_stamped[j].pose.orientation.w 
                q = [qx, qy, qz, qw]
                self.link_orientation[resp.fk_link_names[j]] = euler_from_quaternion(q)
            
            # print(f'Joint positions in Cartesian spapce: {self.link_poses}')
            self.joint_poses.append(positions)
        return self.joint_poses

    def wrap_angle(self, theta):
        return [(t + np.pi) % (2 * np.pi) - np.pi for t in theta]

