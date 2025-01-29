#!/usr/bin/env python3

import time
import rospy 
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest, GetStateValidityResponse
from moveit_msgs.msg import RobotState

DEFAULT_SV_SERVICE = "/check_state_validity"

class StateValidity():
    def __init__(self):
        #rospy.loginfo("Initializing StateValidity class")
        self.sv_srv = rospy.ServiceProxy(DEFAULT_SV_SERVICE, GetStateValidity)
        #rospy.loginfo("Connecting to State Validity Service")
        #rospy.wait_for_service("check_state_validity")
        #rospy.loginfo("Ready for making Validity calls")

        # Prepare msg to interface with MoveIt!
        self.rs = RobotState()
        self.rs.joint_state.name = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
                                    'wrist_3_joint', 'robotiq_85_left_knuckle_joint', 'robotiq_85_right_knuckle_joint', 'robotiq_85_left_inner_knuckle_joint',
                                    'robotiq_85_right_inner_knuckle_joint', 'robotiq_85_left_finger_tip_joint', 'robotiq_85_right_finger_tip_joint']
        self.rs.joint_state.position = [0.0, -1.5447, 1.5447, -1.5794, -1.5794, 0.0, 0.802705053238722, 0.802705053238722,
                                        0.802705053238722, 0.802705053238722, -0.802705053238722, -0.802705053238722]       # Gripper is closed for moving. In manipulation, we can chage that.
        
        self.joint_state_received = False 

    def close_SV(self):
        self.sv_srv.close()

    
    def GetStateValidity(self, robot_state, group_name='ur5e', constraints=None, print_depth=False):
        '''
        Given a RobotState and a group name and an optional constraints
        Returns the validity of the state.
        '''

        # Replace the first six elements of the joint state
        self.rs.joint_state.position[0] = robot_state[0]
        self.rs.joint_state.position[1] = robot_state[1]
        self.rs.joint_state.position[2] = robot_state[2]
        self.rs.joint_state.position[3] = robot_state[3]
        self.rs.joint_state.position[4] = robot_state[4]
        self.rs.joint_state.position[5] = robot_state[5]

        gsvr = GetStateValidityRequest()
        gsvr.robot_state = self.rs 
        gsvr.group_name = group_name

        if constraints != None:
            gsvr.constraints = constraints
        result = self.sv_srv.call(gsvr)

        '''
        if (not result.valid):
            contact_depths = []
            for i in range(len(result.contacts)):
                contact_depths.append(result.contacts[i].depth)
            max_depth = max(contact_depths)
            if max_depth < 0.0001:
                return True
            else:
                return False
        '''

        return result.valid

def IsInCollision(state, col_times=None):
    sv = StateValidity()

    col_start = time.time()
    collision_free = sv.GetStateValidity(state)
    col_end = time.time()

    if col_times is not None:
        col_time = col_end - col_start
        col_times.append(col_time)
        return (not collision_free), col_times

    return (not collision_free)