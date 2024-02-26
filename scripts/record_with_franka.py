import microphone
import microphone_utils
from franka_motion import FrankaMotion
import os
import time
import numpy as np
import sys

import util_datalogging
#create instance of microphone class
devicelist=[0,1,11,12,13,14]
number_of_mics = len(devicelist)
fs = 44100
channels_in = 1

record_duration = 2
wait_duration = record_duration

#create a folder to save data
save_path_data = "/home/iam-lab/audio_localization/audio_datacollection/data/franka_init_test_6mic/"
os.makedirs(save_path_data, exist_ok=True)


def record_with_datalogger(record_duration):
    #init data logger
    data_logger = util_datalogging.DataLogger()
    q_t, qdot_t, ee_pose_t = data_logger.log_q_and_q_d(record_duration) 
    # print(f"len of q_t: {len(q_t)}, qdot_t: {len(qdot_t)}, ee_pose_t: {len(ee_pose_t)}")
    # recorded_q_trajectory.append([q_t, qdot_t])
    # recorded_ee_trajectory.append(ee_pose_t)

    return q_t, qdot_t, ee_pose_t


def main():
    print(f" ------ starting script ------  ")

    

    #init franka robot
    franka_robot = FrankaMotion()
    franka_robot.go_to_init_pose()
    
    franka_robot.go_to_init_recording_pose()

    
    
    trial_count = 0

    #get inital recording x,y position
    initial_recording_pose = franka_robot.get_ee_pose()
    init_x, init_y = initial_recording_pose.translation[0], initial_recording_pose.translation[1]

    #distance along cylinder in 10 evenly spaced intervals from [0 to 20.3]
    distance_sample_count = 100
    cylinder_length = 0.203
    x_along_cylinder = np.linspace(init_x, init_x+cylinder_length, distance_sample_count)

    #save robot joint before traversing across the cylinder distance
    robot_joints_restore_position = franka_robot.get_joints()

    recorded_ee_trajectory = []
    recorded_q_trajectory = []

    x_t_list = [] #  [(200xT, 3), ..., (200xT, 3)]
    xdot_t_list = [] # [(200xT, 3), ..., (200xT, 3)]
    x_t_des_list = [] # [(200xT, 3), ..., (200xT, 3)]
    xdot_t_des_list = [] # [(200xT, 3), ..., (200xT, 3)]


    for i in range(distance_sample_count):

        #get ground truth label [distance along cylinder, joint 6]

        #current pose x - init pose x
        distance = x_along_cylinder[i] #[0.45, 0.45+0.203] robot EE global --> 
        gt_label = [distance-init_x,0] #[0, 0.203] along cylinder

        # move robot 
        franka_robot.move_along_pipe(x_along_cylinder[i])

        # execute tap motion
        # franka_robot.tap_stick(-0.04, duration=record_duration/2)
        franka_robot.tap_stick_joint(duration=record_duration/2)

        mic = microphone.Microphone(devicelist, fs, channels_in)
        mic.record_all_mics(save_path=save_path_data, duration=record_duration, trial_count=trial_count, gt_label=gt_label)

        #record robot state during motion
        # record_with_datalogger(record_duration) #not using this method (ROS subscriber method is not working properly) 
        # x_t, xdot_t, x_t_des, xdot_t_des = franka_robot.record_trajectory(duration=record_duration, dt=0.01) 
        time.sleep(1.5) #wait until skill is sufficiently executed
        # x_t_list.append(x_t)
        # xdot_t_list.append(xdot_t)
        # x_t_des_list.append(x_t_des)
        # xdot_t_des_list.append(xdot_t_des)


        

        # time.sleep(wait_duration/2) #wait for mic to finish recording before calling next skill
        franka_robot.move_away_from_stick_joint(duration=record_duration/2)

        #increment trial count
        trial_count += 1

        #print only up to 4 decimal places
        print(f" trial {i}/{distance_sample_count-1}")

    
    # np.save(f"{save_path_data}recorded_x_trajectory.npy", x_t_list, allow_pickle=True)
    # np.save(f"{save_path_data}recorded_xdot_trajectory.npy", xdot_t_list, allow_pickle=True)
    # np.save(f"{save_path_data}recorded_x_des_trajectory.npy", x_t_des_list, allow_pickle=True)
    # np.save(f"{save_path_data}recorded_xdot_des_trajectory.npy", xdot_t_des_list, allow_pickle=True)

    #print shape of recorded_q_trajectory, recorded_ee_trajectory, gt_contacts
    # print(f"shapes: {len(recorded_q_trajectory), len(recorded_ee_trajectory), len(gt_contacts)}") #--> (distance_sample_count)
    # print(f"qt shapes: {len(recorded_q_trajectory[0][0])}, qdot_t shapes: {len(recorded_q_trajectory[0][1])}")

    # #save robot state 
    # np.save(f"{save_path_data}recorded_q_trajectory.npy", recorded_q_trajectory, allow_pickle=True)
    # np.save(f"{save_path_data}recorded_ee_trajectory.npy", recorded_ee_trajectory, allow_pickle=True)

        

    
    #restore robot to initial position
    print(f"restoring robot to initial cartesian position")
    franka_robot.go_to_init_recording_pose()
    print(f"restoring robot to initial joint position")
    franka_robot.reset_joints()
        

    print(f" ------ ending script ------")




if __name__ == '__main__':
    main()