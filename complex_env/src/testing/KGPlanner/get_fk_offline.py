#!/usr/bin/env python3

import rospy
import numpy as np
import os

from moveit_msgs.srv import GetPositionFK
from moveit_msgs.srv import GetPositionFKRequest
from moveit_msgs.srv import GetPositionFKResponse
from sensor_msgs.msg import JointState

from tf.transformations import euler_from_quaternion

"""
Class to make FK calls using the /compute_fk service.
"""

class GetFK(object):
    def __init__(self, test_data_path=None)->None:
        """
        A class to do FK calls thru the MoveIt!'s /compute_ik service.

        :param str fk_link: link to compute the forward kinematics
        :param str frame_id: frame_id to compute the forward kinematics
        into account collisions
        """
        # rospy.init_node('get_joint_position', anonymous=True)
        self.rate = rospy.Rate(0.1)

        #rospy.loginfo("Initalizing GetFK...")
        self.fk_link = ['shoulder_link', 'upper_arm_link', 'forearm_link', 'wrist_1_link', 'wrist_2_link', 'wrist_3_link']
        self.frame_id = 'base_link'
        rospy.loginfo("PoseStamped answers will be on frame: " + self.frame_id)

        self.fk_srv = rospy.ServiceProxy('/compute_fk', GetPositionFK)
        #rospy.loginfo("Waiting for /compute_fk service...")
        self.fk_srv.wait_for_service()
        #rospy.loginfo("Connected!")

        self.joint_states = [] # for path

        self.link_poses = {}
        self.link_orientation = {}
        self.joint_poses = []
        self.directory = test_data_path

    def get_joint_states(self, trajectory: list)->None:
        '''
        Gets each joint in the trajectory
        '''

        for i in range(len(trajectory)):
            position = trajectory[i]
            joint_state = JointState()
            joint_state.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
            joint_state.position = position
            self.joint_states.append(joint_state)

    def get_joint_state(self, joint: list)->None:
        '''
        Get one joint coordinates in 3D
        '''
        joint_state = JointState()
        joint_state.name = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        joint_state.position = joint
        self.joint_states = joint_state
        return joint_state

    def get_current_fk(self, joint_idx: int)->None:
        '''
        Forward Kinematics
        '''
        while not rospy.is_shutdown() and self.joint_states[joint_idx] is None:
            rospy.logwarn("Waiting for a /joint_states message...")
            self.rate.sleep()
        return self.get_fk(joint_idx)

    def get_current_fk_joint(self, joint_state: list)->None:
        '''
        Forward kinematics for a joint
        '''
        while not rospy.is_shutdown() and joint_state is None:
            rospy.logwarn("Waiting for a /joint_state message ...")
            self.rate.sleep()
        return self.get_fk_joint(joint_state)

    def get_fk(self, joint_idx: int):
        """
        Do an FK call to with.

        :param sensor_msgs/JointState joint_state: JointState message
            containing the full state of the robot.
        :param str or None fk_link: link to compute the forward kinematics for.
        """
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
            rospy.logerr("Service exception: " + str(e))
            resp = GetPositionFKResponse()
            resp.error_code = 99999  # Failure
            return resp

    def get_fk_joint(self, joint_state):
        '''
        Do an FK call to a point in the configuration space.
        '''

        if self.fk_link is None:
            fk_link = self.fk_link

        req = GetPositionFKRequest()
        req.header.frame_id = self.frame_id
        req.fk_link_names = self.fk_link
        req.robot_state.joint_state = joint_state
        try:
            resp = self.fk_srv.call(req)
            return resp

        except rospy.ServiceException as e:
            rospy.logerr("Service exception: " + str(e))
            rospy.GetPositionFKResponse()
            resp.error_code = 99999                      # Failure
            return resp
        
    def run_path(self, trajectory: list)-> list:
        '''
        Forward Kinematics Service is being performed for a whole path
        '''
        self.get_joint_states(trajectory)

        for i in range(len(self.joint_states)):
            resp = self.get_current_fk(i)
            i = len(resp.pose_stamped)
            positions = []
            for i in range(len(resp.pose_stamped)):
                x = resp.pose_stamped[i].pose.position.x
                y = resp.pose_stamped[i].pose.position.y
                z = resp.pose_stamped[i].pose.position.z

                position = [x, y, z]
                positions.append(position)
                self.link_poses[resp.fk_link_names[i]] = position

                qx = resp.pose_stamped[i].pose.orientation.x
                qy = resp.pose_stamped[i].pose.orientation.y
                qz = resp.pose_stamped[i].pose.orientation.z
                qw = resp.pose_stamped[i].pose.orientation.w
                q = [qx, qy, qz, qw]
                self.link_orientation[resp.fk_link_names[i]] = euler_from_quaternion(q)

            #print(f'Joint Positions in Cartesian Space: {self.link_poses}')
            self.joint_poses.append(positions)

        file_name = os.path.join(self.directory, 'traj_RRTstar_raw_fk_1')
        np.save(file_name, self.joint_poses)
        return self.joint_poses

    def run_joint(self, joint: list)-> list:
        '''
        Forward kinematics service is being performed for one joint
        '''

        joint_state = self.get_joint_state(joint)

        resp = self.get_current_fk_joint(joint_state)
        positions = []
        for i in range(len(resp.pose_stamped)):
            x = resp.pose_stamped[i].pose.position.x
            y = resp.pose_stamped[i].pose.position.y
            z = resp.pose_stamped[i].pose.position.z

            position = [x, y, z]
            positions.append(position)

            qx = resp.pose_stamped[i].pose.orientation.x
            qy = resp.pose_stamped[i].pose.orientation.y
            qz = resp.pose_stamped[i].pose.orientation.z
            qw = resp.pose_stamped[i].pose.orientation.w
            q = [qx, qy, qz, qw]
            self.link_orientation[resp.fk_link_names[i]] = euler_from_quaternion(q)

        return positions



