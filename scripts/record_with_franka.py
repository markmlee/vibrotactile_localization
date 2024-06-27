import microphone
from franka_motion import FrankaMotion
import os
import time
import numpy as np
import sys

import threading

# import util_datalogging

import rospy
from geometry_msgs.msg import Point

# ================= PARAMS FOR EACH RUN ===================================================================================
data_recording = True # used when running eval for online inference only. TURN ON for offline data collection.

# devicelist=[2,10,11,12,13,14]
# number_of_mics = len(devicelist)

devicelist=[2]
number_of_mics = 6
channels_in = 6
fs = 44100


record_duration = 2 #seconds for audio, trajectory recording
wait_duration = record_duration
distance_sample_count = 10 #20 number of samples to take along the height (1cm res)
radian_sample_count = 10  #10 number of samples to take along the radian (300 degrees/10 = 30 degrees res)
total_repeat_count = 2 #3 number of times to repeat the 2D motion (3 times)
#--> should get total number of 2400 trials (20x10x3x4 = 2400)

goal_j1_angle_min, goal_j1_angle_max = 14, 16 #12,15 for non-angled #[7.0 MIN, 15 MAX] #degrees for tap stick joint
j7_radian_min, j7_radian_max =  -2.7, 2.7 #0,0 #-2.7, 2.7 #radians for tap stick joint
tap_angle_min, tap_angle_max = 0, 0 #0, 10 #degrees for tap stick joint ****************** CHANGE HERE ************************
HIT_AT_ANGLE = False #****************** CHANGE HERE ************************
if HIT_AT_ANGLE:
    distance_sample_count = 10 #20 number of samples to take along the height (1cm res)
    radian_sample_count = 10 #number of samples to take along the radian (300 degrees/10 = 30 degrees res)
    total_repeat_count = 2 #number of times to repeat the 2D motion (3 times)
    tap_angle_min, tap_angle_max = 0, 10 #0, 10 #degrees for tap stick joint ****************** CHANGE HERE ************************



y_location_min, y_location_max = 0.25, 0.40 #meters along the y-axis to traverse from robot base
y_location_sample_count = 3
y_location_list = np.linspace(y_location_min, y_location_max, y_location_sample_count)


save_path_data = "/home/iam-lab/audio_localization/vibrotactile_localization/data/wood_T50_L42_Horizontal/"

RADIAN_HARDCODE_EVAL = False #TODO ****************** CHANGE HERE ************************

#radian angle in 5 evenly spaced intervals from [-2.8 to 2.8]
radian_along_cylinder = np.linspace(j7_radian_min, j7_radian_max, radian_sample_count)
if RADIAN_HARDCODE_EVAL:
    y_location_min, y_location_max = 0.40, 0.40 #meters along the y-axis to traverse from robot base
    save_path_data = "/home/iam-lab/audio_localization/vibrotactile_localization/data/test_generalization/stick_T50_L42_Y_40/"

    distance_sample_count = 5 #number of samples to take along the height
    total_repeat_count = 1 #number of times to repeat the 2D motion
    radian_along_cylinder = [-2.7, -1.54285714, 0, 1.54285714, 2.7]
    radian_sample_count = len(radian_along_cylinder)
    tap_angle_min, tap_angle_max = 0,0 #no tilt when hitting the stick
    goal_j1_angle_min, goal_j1_angle_max = 15, 15 #no varying speed when hitting the stick
    tap_angle_min, tap_angle_max = 0, 0 #degrees for tap stick joint
    HIT_AT_ANGLE = False

    
    y_location_sample_count = 1
    y_location_list = np.linspace(y_location_min, y_location_max, y_location_sample_count)


total_trial_count = 0
end_trial_count = distance_sample_count * radian_sample_count * total_repeat_count * y_location_sample_count
print(f" *********** Expected total trials: {end_trial_count} ***********")


cylinder_length = 0.203 #meters along the cylinder to traverse

#create a folder to save data
# save_path_data = "/home/iam-lab/audio_localization/vibrotactile_localization/data/franka_2D_localization_online_run/"
os.makedirs(save_path_data, exist_ok=True)
# ====================================================================================================


def scale_j1_angle_with_y(y_hit_location, goal_j1_angle_min, goal_j1_angle_max):
    """
    scale the j1 angle with the y location value (linear interpolated between min and max) [12,15] at y=0.25, [10,12] at y=0.40
    so that smaller y has bigger angle and vice versa.
    Has the effect of making impact similar for each y location
    """
    angle_min_ = 12 #10
    angle_max_ = 14 #12 #j1 angle to scale with y location

    theta_min = goal_j1_angle_min +  (angle_min_ - goal_j1_angle_min)/(0.4-0.25) * (y_hit_location - 0.25)
    theta_max = goal_j1_angle_max + (angle_max_ - goal_j1_angle_max)/(0.4-0.25) * (y_hit_location - 0.25)
    
    random_angle_scaled = np.random.uniform(theta_min, theta_max)

    return random_angle_scaled


def tap_along_1D_cylinder(franka_robot, z_along_cylinder, init_z, current_ee_RigidTransform_rotm, y_hit_location, gt_label_rad, pub_contactloc):
    """
    motion to traverse along the cylinder in 1D line.
    record audio and traj data at each point
    """
    global total_trial_count, distance_sample_count, cylinder_length

    x_t_list = []
    xdot_t_list = []
    x_t_des_list = []
    xdot_t_des_list = []
    q_t_list = []
    q_tau_list = []


    for i in range(distance_sample_count):

        #get ground truth label [distance along cylinder, joint 6]

        #current pose x - init pose x
        distance = z_along_cylinder[i] #[0.45, 0.45+0.203] robot EE global --> 
        gt_label = [init_z-distance - cylinder_length/2 ,gt_label_rad] #[-0.101, 0,  +0.101] height, and [-2.7, 2.7] radian

        print(f" ---- #5. moving to {distance} along the cylinder ----")
        franka_robot.move_with_fixed_orientation(x=0.12, y=y_hit_location, z= z_along_cylinder[i], current_ee_RigidTransform_rotm = current_ee_RigidTransform_rotm)

        # store joint position
        joints_before_contact = franka_robot.get_joints() 

        if HIT_AT_ANGLE:
            # random tap angle degree from [-10,0]
            tap_angle = np.random.uniform(-tap_angle_max, tap_angle_min)
            print(f" +++++++++ #5. rotating EE orientation by {tap_angle} degrees +++++++++ ")
            franka_robot.rotate_ee_orientation(tap_angle)
            time.sleep(3) #wait until skill is sufficiently executed
        
        #randomly choose an angle given the min,max range BUT scale with Y location value
        goal_j1_angle = scale_j1_angle_with_y(y_hit_location, goal_j1_angle_min, goal_j1_angle_max)

        #publish Gt label for evaluation
        # contact_pt = Point()
        # contact_pt.x = np.cos(gt_label[1])
        # contact_pt.y = np.sin(gt_label[1])
        # contact_pt.z = gt_label[0]*100 #convert to cm
        # pub_contactloc.publish(contact_pt)

        print(f" ---- #6. tapping stick joint at {goal_j1_angle} ----")
        franka_robot.tap_stick_joint(duration=record_duration/2, goal_j1_angle = goal_j1_angle)

        if data_recording:
            # Create a Thread object and start it
            mic = microphone.Microphone(devicelist, fs, channels_in)
            record_mics_thread = threading.Thread(target=mic.record_all_mics, args=(save_path_data, record_duration, total_trial_count, gt_label))
            record_mics_thread.start()

        #get proprioceptive data
        x_t, xdot_t, x_t_des, xdot_t_des, q_t, q_tau = franka_robot.record_trajectory(duration=record_duration, dt=0.01) 
        

        #save the recorded trajectory in save_path_data
        franka_robot.save_recorded_trajectory(save_path_data, total_trial_count , x_t, xdot_t, x_t_des, xdot_t_des, q_t, q_tau, goal_j1_angle)

        #create a folder for each trial
        save_folder_path = f"{save_path_data}trial{total_trial_count}/"
        #save ground truth label to folder as npy file
        np.save(f"{save_folder_path}gt_label.npy", gt_label)


        time.sleep(2.5) #wait until skill is sufficiently executed

        # restore joint positions after tap
        franka_robot.go_to_joint_position(joints_before_contact, duration=2)

        #increment trial count
        total_trial_count += 1

        #print only up to 4 decimal places
        print(f" trial {i}/{distance_sample_count-1}. Current status trial: {total_trial_count}/{end_trial_count}")

        #append data to list
        x_t_list.append(x_t)
        xdot_t_list.append(xdot_t)
        x_t_des_list.append(x_t_des)
        xdot_t_des_list.append(xdot_t_des)
        q_t_list.append(q_t)
        q_tau_list.append(q_tau)

    return x_t_list, xdot_t_list, x_t_des_list, xdot_t_des_list, q_t_list, q_tau_list

def main():
    global radian_sample_count

    print(f" ------ starting script ------  ")

    

    #init franka robot
    franka_robot = FrankaMotion()

    #create ros publisher for contact location GT
    pub_contactloc = rospy.Publisher('/contact_location_GT', Point, queue_size=10)


    print(f" ===== #1. go to initial pose =====")
    franka_robot.go_to_init_pose()
    print(f" ===== #2. go to initial recording pose =====")
    franka_robot.go_to_init_recording_pose()

    
    #get inital recording x,y position
    initial_recording_pose = franka_robot.get_ee_pose()
    init_y, init_z = initial_recording_pose.translation[1], initial_recording_pose.translation[2]

    #distance along cylinder in 10 evenly spaced intervals from [0 to 20.3]
    z_along_cylinder = np.linspace(init_z, init_z-cylinder_length, distance_sample_count)

    

    

    #save robot joint before traversing across the cylinder distance
    robot_joints_restore_position = franka_robot.get_joints()


    x_t_list_2D = [] #  [(200xT, 3), ..., (200xT, 3)]
    xdot_t_list_2D = [] # [(200xT, 3), ..., (200xT, 3)]
    x_t_des_list_2D = [] # [(200xT, 3), ..., (200xT, 3)]
    xdot_t_des_list_2D = [] # [(200xT, 3), ..., (200xT, 3)]
    q_t_list_2D = [] # [(200xT, 7), ..., (200xT, 7)]
    q_tau_list_2D = [] # [(200xT, 7), ..., (200xT, 7)]

    for repeat in range(total_repeat_count):
        
        print(f" ************** repeat: {repeat}/{total_repeat_count} total repeats **************")
        print(f" ===== #3N. go to initial recording pose =====")
        franka_robot.go_to_init_recording_pose(duration=10)

        for y_location_index, y_location_val in enumerate(y_location_list):

            y_hit_location = y_location_val

            #--------------------------- data motion to traverse along the cylinder 2D [height, radian] ---------------------------
            for i in range(radian_sample_count):

                print(f" i: {i} / {radian_sample_count} total radian samples")

                #go to initial joint position
                franka_robot.go_to_joint_position(robot_joints_restore_position, duration=10)

                #go to j7 angle in array
                print(f"#4. rotating j7 to {radian_along_cylinder[i]}")
                franka_robot.rotate_j7(radian_along_cylinder[i])
                gt_label_rad = radian_along_cylinder[i]

                #get current ee quaternion
                current_ee_RigidTransform_rotm = franka_robot.get_ee_pose().rotation

                #--------------------------- record trajectory and audio at each point along the cylinder ---------------------------
                x_t, xdot_t, x_t_des, xdot_t_des, q_t, q_tau = tap_along_1D_cylinder(franka_robot, z_along_cylinder, init_z, current_ee_RigidTransform_rotm, y_hit_location, gt_label_rad, pub_contactloc)

                x_t_list_2D.extend(x_t)
                xdot_t_list_2D.extend(xdot_t)
                x_t_des_list_2D.extend(x_t_des)
                xdot_t_des_list_2D.extend(xdot_t_des)
                q_t_list_2D.extend(q_t)
                q_tau_list_2D.extend(q_tau)

                print(f"rad: {i+1} / {radian_sample_count} , y_loc: {y_location_index+1} / {len(y_location_list)}, repeat: {repeat+1}/{total_repeat_count}")

            

    np.save(f"{save_path_data}recorded_x_trajectory.npy", x_t_list_2D, allow_pickle=True)
    np.save(f"{save_path_data}recorded_xdot_trajectory.npy", xdot_t_list_2D, allow_pickle=True)
    np.save(f"{save_path_data}recorded_x_des_trajectory.npy", x_t_des_list_2D, allow_pickle=True)
    np.save(f"{save_path_data}recorded_xdot_des_trajectory.npy", xdot_t_des_list_2D, allow_pickle=True)
    np.save(f"{save_path_data}recorded_q_trajectory.npy", q_t_list_2D, allow_pickle=True)
    np.save(f"{save_path_data}recorded_q_tau_trajectory.npy", q_tau_list_2D, allow_pickle=True)

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