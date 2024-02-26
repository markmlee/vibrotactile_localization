import microphone
import microphone_utils
# from franka_motion import FrankaMotion
import argparse

import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import Float64MultiArray
import time

import franka_interface_msgs.msg
from franka_interface_msgs.msg import RobotState
import os
"""
This script subscribes to various topics {robot state, microphone, image} and logs the data to a file to be plotted later.
"""

class DataLogger:
    def __init__(self):
        self.init_data()

        # rospy.init_node('data_logger_node', anonymous=True)

    def q_callback(self, msg):
        # print(f"q_callback: {msg.q}")
        self.q_data.append(msg.q)
        self.q_d_data.append(msg.q_d)
        self.ee_pose_data.append(msg.O_T_EE) #4x4 matrix


    def init_data(self):
        self.q_data = []
        self.q_d_data = []
        self.ee_pose_data = []

    def log_q_and_q_d(self, duration):
        
        #clear the data
        self.q_data.clear()
        self.q_d_data.clear()
        self.ee_pose_data.clear()

        rospy.Subscriber("/robot_state_publisher_node_1/robot_state", RobotState, self.q_callback)
        start_time = rospy.get_time()
        rate = rospy.Rate(100)  # 100 Hz

        while rospy.get_time() - start_time < duration:
            # print(f"time elapsed: {rospy.get_time() - start_time}")
            rate.sleep()

        return self.q_data, self.q_d_data, self.ee_pose_data

q_list = []
q_d_list = []
ee_pose_list = []

def state_callback(msg):
    # print(f"q: {msg.q}")
    # print(f"q_d: {msg.q_d}")
    # print(f"O_T_EE: {msg.O_T_EE}")
    q_list.append(msg.q)
    q_d_list.append(msg.q_d)
    ee_pose_list.append(msg.O_T_EE)

def main():
    print(f" ------ starting script ------  ")

    # data_logger = DataLogger()
    # q_data, q_d_data = data_logger.log_q_and_q_d(3)  # replace 10 with actual duration

    #create ros node
    rospy.init_node('data_logger_node', anonymous=True)

    #ros rate
    rate = rospy.Rate(1)  # 100 Hz

    #create a ros subscriber to robot state to record q, q_d, ee_pose
    rospy.Subscriber("/robot_state_publisher_node_1/robot_state", RobotState, state_callback)

    while not rospy.is_shutdown():
        rate.sleep()

    #save q, q_d, ee_pose
    path_name = "/home/iam-lab/audio_localization/audio_datacollection/data/franka_init_test_6mic/"
    filenameq = os.path.join(path_name, "recorded_q.npy")
    filenameq_d = os.path.join(path_name, "recorded_q_d.npy")
    filenameee_pose = os.path.join(path_name, "recorded_ee_pose.npy")
    
    np.save(filenameq, q_list)
    np.save(filenameq_d, q_d_list)
    np.save(filenameee_pose, ee_pose_list)

    print(f" ------ ending script ------")

if __name__ == '__main__':
    
    main()