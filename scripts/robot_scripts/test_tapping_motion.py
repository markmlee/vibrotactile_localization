import os
import time
import numpy as np
import sys
import torch
from tqdm import tqdm
import math

sys.path.append(os.path.abspath(os.path.join('../')))
from franka_motion import FrankaMotion

import microphone
import threading

from std_msgs.msg import String, Int32
from geometry_msgs.msg import Point

#ros
import rospy
from visualization_msgs.msg import Marker, MarkerArray
#hydra
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig


#custom models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../learning')))
from models.CNN import CNNRegressor2D

#dataset
from datasets import AudioDataset

#custom utils
import microphone_utils as mic_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../learning')))
from transforms import to_mel_spectrogram, get_signal

# ================================================================================================================
ROBOT_MOTION = True
PREDICT_MODEL = True
data_recording = True # used when running eval for online inference only. TURN ON for offline data collection.
save_path_data = '/home/iam-lab/audio_localization/vibrotactile_localization/data/test_mapping/T25L42_Horizontal_SingleStick/'
devicelist=[2]
number_of_mics = 6
channels_in = 6
fs = 44100

cylinder_length = 0.203 #meters along the cylinder to traverse
goal_j1_angle = 14
record_duration = 2
total_trial_count = 0
num_samples = 4
radian_sample_count = 4
j7_radian_min, j7_radian_max = -2.7, 2.7 #0,0 #-2.7, 2.7 #radians for tap stick joint
radian_along_cylinder = np.linspace(j7_radian_min, j7_radian_max, radian_sample_count)

# ================================================================================================================

def tap_along_y_axis_line(franka_robot, y_along_woodenstick, init_z,  current_ee_RigidTransform_rotm, gt_label_rad):
    """
    motion to tap along the y-axis line. Traverse from start_y to end_y by taking num_samples samples.
    """
    global goal_j1_angle, record_duration, data_recording, save_path_data, total_trial_count, data_recording
    num_samples = len(y_along_woodenstick)

    for sample in range(num_samples):

        distance = y_along_woodenstick[sample]

        gt_label = [-0.101 ,gt_label_rad] #[-0.101, 0,  +0.101] height, and [-2.7, 2.7] radian
        print(f"gt label: {gt_label}")

        print(f" ---- #5. moving to {distance} along the cylinder ----")
        franka_robot.move_with_fixed_orientation(x=0.10, y=y_along_woodenstick[sample], z= init_z, current_ee_RigidTransform_rotm = current_ee_RigidTransform_rotm)

        # store joint position
        joints_before_contact = franka_robot.get_joints() 

        print(f" ---- #6. tapping stick joint at {goal_j1_angle} ----")
        franka_robot.tap_stick_joint(duration=record_duration/2, goal_j1_angle = goal_j1_angle)

        

        

        if data_recording:
            # Create a Thread object and start it
            mic = microphone.Microphone(devicelist, fs, channels_in)
            record_mics_thread = threading.Thread(target=mic.record_all_mics, args=(save_path_data, record_duration, total_trial_count, gt_label))
            record_mics_thread.start()

        time.sleep(3)
        
        #increment trial count
        total_trial_count += 1



def test_blind_ee_orientation_motion(franka_robot):
    """
    for debugging purposes, test the motion of rotating the ee orientation without any tapping motion
    """
    
    print(f" ===== #1.reset joints =====")
    franka_robot.reset_joints()

    print(f" ===== #2. run the ee orientation =====")
    # franka_robot.rotate_ee_orientation()

    T_ee_current_pose = franka_robot.franka.get_pose()

    print(f"T_ee_current_pose {T_ee_current_pose}")

    franka_robot.verify_motion_rotate_ee_orientation()



def calculate_radian_error(rad_pred, rad_val):
    """
    calculate the degree error between the predicted and ground truth radian values
    This resolves wrap around issues
    """
    #diff = pred - GT
    #add +pi
    #mod by 2pi
    #subtract pi

    # print(f"0. rad_pred: {rad_pred}, rad_val: {rad_val}")
    rad_diff = rad_pred - rad_val
    # print(f"1. rad_diff: {rad_diff}")
    rad_diff = rad_diff + math.pi
    # print(f"2. rad_diff: {rad_diff}")
    rad_diff = torch.remainder(rad_diff, 2*math.pi)
    # print(f"3. rad_diff: {rad_diff}")
    radian_error = rad_diff - math.pi
    # print(f"4. radian_error: {radian_error}")

    return radian_error

def get_hitting_location(rod_start_position, rod_end_position, number_of_hitting_samples, stick_length, stick_thickness, stick_tapping_offset, stick_axis):
    """
    input:
    rod_start_position: [x1,y1,z1] #start hitting position of rod
    rod_end_position: [x2,y2,z2]   #end hitting position of rod
    number_of_hitting_samples:      number of hitting samples along the rod
    stick_length:                   length of the stick
    stick_thickness:                thickness of the stick
    stick_tapping_offset:           offset from the stick to get hitting position
    stick_axis:                     axis along which the stick is oriented, different from robot hitting axis

    output:
    hitting_location_list:          list of hitting locations along the stick [ [x,y,z]1...[x,y,z]N]
    """

    #given the stick axis, create xyz points with number_of_hitting_samples
    if stick_axis == 'y':
        x_stick_pos = np.linspace(rod_start_position[0], rod_end_position[0], number_of_hitting_samples)
        y_stick_pos = np.linspace(rod_start_position[1], rod_end_position[1], number_of_hitting_samples)
        z_stick_pos = np.linspace(rod_start_position[2], rod_end_position[2], number_of_hitting_samples)

        x_hitting_pos = x_stick_pos +  stick_thickness/2 + stick_tapping_offset # no change
        y_hitting_pos = y_stick_pos #varies
        z_hitting_pos = z_stick_pos # no change

        x_hitting_pos_opposite = x_stick_pos - stick_thickness/2 - stick_tapping_offset # no change
        y_hitting_pos_opposite = y_stick_pos
        z_hitting_pos_opposite = z_stick_pos

    hitting_location_list = []
    hitting_location_opposite_list = []

    #add one side of hitting location first
    for i in range(number_of_hitting_samples):
        hitting_location_list.append([x_hitting_pos[i], y_hitting_pos[i], z_hitting_pos[i]])
    #add opposite side of hitting location second
    for i in range(number_of_hitting_samples):
        hitting_location_opposite_list.append([x_hitting_pos_opposite[i], y_hitting_pos_opposite[i], z_hitting_pos_opposite[i]])

    return hitting_location_list, hitting_location_opposite_list

def predict_contact_from_wavfile(cfg):
    """
    input: trial_directory which contains mic2.wav
    output: predicted height and radian from the model
    """

    #create dataloader
    #load data
    dataset = AudioDataset(cfg=cfg, data_dir = cfg.data_dir, transform = cfg.transform, augment = False)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.val_batch_size, shuffle=False) 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    #load model.pth from checkpoint
    model = CNNRegressor2D(cfg)
    model.load_state_dict(torch.load(os.path.join(cfg.model_directory, 'model.pth')))

    #verify if model is loaded by checking the model parameters
    # print(model)
    model.to(device)
    model.eval()

    #for item in data_loader:
    for _, (x, y, _) in enumerate(tqdm(val_loader)):
        x_input, Y_val = x.float().to(device), y.float().to(device)

        with torch.no_grad():
            Y_output = model(x_input) 

            #split prediction to height and radian
            height_pred = Y_output[:,0]

            #clip height to [-11, +11]
            height_pred = torch.clamp(height_pred, -11, 11)

            x_pred = Y_output[:,1]
            y_pred = Y_output[:,2]

            #clip x and y to [-1, +1]
            x_pred = torch.clamp(x_pred, -1, 1)
            y_pred = torch.clamp(y_pred, -1, 1)

            #convert y_val to radian
            x_val = Y_val[:,1]
            y_val = Y_val[:,2]
            radian_val = torch.atan2(y_val, x_val)

            #convert y_pred to radian
            radian_pred = torch.atan2(y_pred, x_pred)
            deg_pred = torch.rad2deg(radian_pred)

            #resolve wrap around angle issues
            radian_error = calculate_radian_error(radian_pred, radian_val)
            degree_diff = torch.rad2deg(radian_error)


    print(f"height_pred: {height_pred}, deg_pred: {deg_pred},  degree_diff: {degree_diff}")

    return height_pred, x_pred, y_pred

def create_marker_and_publish(franka_robot, pub_contactloc, contact_pt, total_trial_count, pub_stick_markerarray, stick_markerarray):
    scale_xyz_list = [0.05, 0.05, 0.05]
    color_argba_list = [1, 1, 0, 0]
    pred_marker = franka_robot.create_marker(total_trial_count, contact_pt,  scale_xyz_list, color_argba_list, lifetime=2)
    
    #publish the predicted contact point
    pub_contactloc.publish(pred_marker)

    #create new marker for stick array
    array_scale_xyz_list = [0.03, 0.03, 0.03]
    array_color_argba_list = [1, 0, 1, 0]
    array_marker = franka_robot.create_marker(total_trial_count, contact_pt,  array_scale_xyz_list, array_color_argba_list, lifetime=30)

    #publish the stick location
    stick_markerarray.markers.append(array_marker)
    pub_stick_markerarray.publish(stick_markerarray)

def robot_hit_along_list(hitting_location_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, stick_markerarray, opposite_side=False):
    """
    robot execute tapping motion along 1 side of the stick
    hit along the stick and publish
    """
    global total_trial_count

        #iterate through hitting locations
    for count, hit_location in enumerate(hitting_location_list):
        print(f"hit_location: {hit_location}")

        #go to hit location
        print(f" ===== #a. go to hitting location {hit_location} =====")
        if ROBOT_MOTION: franka_robot.move_with_fixed_orientation(x=hit_location[0], y=hit_location[1], z=hit_location[2])

        #store current joint position
        joints_before_contact = franka_robot.get_joints()

        if opposite_side:
            hit_j1_angle = -1 * goal_j1_angle
        else:
            hit_j1_angle = goal_j1_angle

        #tap stick
        print(f" ===== #b. tapping stick joint at {hit_j1_angle} ===== ")
        if ROBOT_MOTION: franka_robot.tap_stick_y_joint(duration=record_duration/2, goal_j1_angle = hit_j1_angle)

        if data_recording:
            # Create a Thread object and start it
            mic = microphone.Microphone(devicelist, fs, channels_in)
            record_mics_thread = threading.Thread(target=mic.record_all_mics, args=(save_path_data, record_duration, total_trial_count, gt_label))
            record_mics_thread.start()

        #get proprioceptive data
        x_t, xdot_t, x_t_des, xdot_t_des, q_t, q_tau = franka_robot.record_trajectory(duration=record_duration, dt=0.01) 
        
        #save the recorded trajectory in save_path_data
        franka_robot.save_recorded_trajectory(save_path_data, total_trial_count , x_t, xdot_t, x_t_des, xdot_t_des, q_t, q_tau, goal_j1_angle)

        time.sleep(1)

        # ************* make model prediction here *************
        if PREDICT_MODEL:
            height_pred, x_pred, y_pred = predict_contact_from_wavfile(cfg)
            height_i, x_i, y_i = height_pred[total_trial_count].item(), x_pred[total_trial_count].item(), y_pred[total_trial_count].item()
            print(f"trial: {total_trial_count}, height_pred: {height_i}, x_pred: {x_i}, y_pred: {y_i}")
        # ****************************************************

        #post process XY -> radian -> XY (to ensure projection is on the unit circle) AND (resolve wrap around issues) 
        contact_pt = franka_robot.transform_predicted_XYZ_to_EE_XYZ(x_i, y_i,height_i, cfg.cylinder_radius, cfg.cylinder_transform_offset)
        #publish makers
        create_marker_and_publish(franka_robot, pub_contactloc, contact_pt, total_trial_count, pub_stick_markerarray, stick_markerarray)


        #restore robot to joint position
        print(f" ===== #c. moving back to precontact joints  ===== ")
        if ROBOT_MOTION: franka_robot.go_to_joint_position(joints_before_contact, duration=5)

        #increment trial count
        total_trial_count += 1

    

@hydra.main(version_base='1.3',config_path='../../learning/configs', config_name = 'inference')
def main(cfg: DictConfig):
    global total_trial_count

    print(f" ------ starting script ------  ")

    #init franka robot
    franka_robot = FrankaMotion()

    # ROS publisher for the contact location
    pub_contactloc = rospy.Publisher('/contact_location', Marker, queue_size=10) #--> publish X: rad_X, Y: rad_Y, Z:height
    pub_stick_markerarray = rospy.Publisher('/stick_location', MarkerArray, queue_size=10)

    print(f" ===== #1. go to initial pose =====")
    if ROBOT_MOTION: franka_robot.go_to_init_pose()
    print(f" ===== #2. go to initial recording pose =====")
    if ROBOT_MOTION: franka_robot.go_to_init_recording_pose()

    robot_joints_restore_position = franka_robot.get_joints()

    print(f" ===== #3. rotate to j7 = 0 =====")
    if ROBOT_MOTION: franka_robot.rotate_j7(0)

    #store information about the cylinder
    #get inital recording x,y position
    initial_recording_pose = franka_robot.get_ee_pose()
    init_y, init_z = initial_recording_pose.translation[1], initial_recording_pose.translation[2]
    j7_joint_radian = franka_robot.get_joints()[6]

    gt_label = [-0.101, j7_joint_radian] #[-0.101 fixed height, 0 fixed radian]
    print(f"gt label: {gt_label}")
    
    x1,y1,z1 = 0, 0.27, 0.57
    x2,y2,z2 = 0, 0.38, 0.57
    rod_start_position = [x1,y1,z1]
    rod_end_position = [x2,y2,z2]
    number_of_hitting_samples = 3
    stick_length = 0.15
    stick_thickness = 0.025
    stick_tapping_offset = 0.1
    stick_axis = 'y'

    hitting_location_list, hitting_location_opposite_list = get_hitting_location(rod_start_position, rod_end_position, number_of_hitting_samples, stick_length, stick_thickness, stick_tapping_offset, stick_axis)
    
    #markerarray for the rod
    stick_markerarray = MarkerArray()

    print(f" ===== #4. Hit along {hitting_location_list} =====")
    robot_hit_along_list(hitting_location_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, stick_markerarray, opposite_side=False)

    #move to opposite side of the stick
    print(f" ===== #5. moving to opposite side of stick =====")
    if ROBOT_MOTION: franka_robot.move_with_fixed_orientation(x=0, y=0.35, z=0.8, current_ee_RigidTransform_rotm = initial_recording_pose.rotation, duration=5)

    print(f" ===== #6. Hit along {hitting_location_opposite_list} =====")
    robot_hit_along_list(hitting_location_opposite_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, stick_markerarray, opposite_side=True)

    #move to opposite side of the stick
    print(f" ===== #7. moving to opposite side of stick =====")
    if ROBOT_MOTION: franka_robot.move_with_fixed_orientation(x=0, y=0.35, z=0.8, current_ee_RigidTransform_rotm = initial_recording_pose.rotation, duration=5)

    #restore robot to home position
    print(f"restoring robot to home joints")
    if ROBOT_MOTION: franka_robot.reset_joints()

    print(f" ------ ending script ------  ")

if __name__ == '__main__':
    main()