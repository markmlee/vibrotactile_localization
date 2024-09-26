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
import tf

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


devicelist=[2]
number_of_mics = 6
channels_in = 6
fs = 44100

# ================================================================================================================
ROBOT_MOTION = True
PREDICT_MODEL = True
data_recording = True # used when running eval for online inference only. TURN ON for offline data collection.
save_path_data = '/home/iam-lab/audio_localization/vibrotactile_localization/data/test_mapping/cross_easy_randomexplore_v1/'


number_of_hitting_samples = 4

cylinder_length = 0.203 #meters along the cylinder to traverse
goal_j1_angle = 15
record_duration = 2
total_trial_count = 0
# num_samples = 4
radian_sample_count = 4
j7_radian_min, j7_radian_max = -2.7, 2.7 #0,0 #-2.7, 2.7 #radians for tap stick joint
radian_along_cylinder = np.linspace(j7_radian_min, j7_radian_max, radian_sample_count)

j7_radian_init = -2.1

output_prediction_list = []
output_avg_prediction_list = []

# ================================================================================================================

def create_marker_and_publish(franka_robot, pub_contactloc, contact_pt, total_trial_count, pub_stick_markerarray, stick_markerarray):
    scale_xyz_list = [0.05, 0.05, 0.05]
    color_argba_list = [1, 1, 0, 0]
    pred_marker = franka_robot.create_marker(total_trial_count, contact_pt,  scale_xyz_list, color_argba_list, lifetime=2, frame_id="cylinder_origin")
    
    #publish the predicted contact point
    pub_contactloc.publish(pred_marker)

    #create new marker for stick array
    array_scale_xyz_list = [0.03, 0.03, 0.03]
    array_color_argba_list = [1, 0, 1, 0]
    array_marker = franka_robot.create_marker(total_trial_count, contact_pt,  array_scale_xyz_list, array_color_argba_list, lifetime=3000, frame_id="cylinder_origin")

    #publish the stick location
    stick_markerarray.markers.append(array_marker)
    pub_stick_markerarray.publish(stick_markerarray)


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


    # print(f"height_pred: {height_pred}, deg_pred: {deg_pred},  degree_diff: {degree_diff}")

    return height_pred, x_pred, y_pred

def tap_all_four_directions(cfg,franka_robot, goal_j1_angle, pub_contactloc, pub_stick_markerarray, stick_markerarray):
    """
    Once arriving at a sample point position, move in all 4 directions using joint motion.
    Horizontal - Left/Right using j1 angle
    Vertical - Up/Down using j6 angle
    """

    global total_trial_count

    num_sides = 4

    for i in range(num_sides):


        #store current joint position
        joints_before_contact = franka_robot.get_joints()

        gt_label = franka_robot.get_manual_gt_label()

        #tapping motion
        if i == 0:
            print(f"tapping RIGHT {i+1}/{num_sides}, go to j1 angle {goal_j1_angle}, trial: {total_trial_count}")
            franka_robot.tap_stick_joint(duration=record_duration/2, goal_j1_angle = goal_j1_angle)

        elif i == 1:
            print(f"tapping LEFT {i+1}/{num_sides}, go to j1 angle {goal_j1_angle*-1}, trial: {total_trial_count}")
            franka_robot.tap_stick_joint(duration=record_duration/2, goal_j1_angle = goal_j1_angle * -1)

        elif i == 2:
            print(f"tapping UP {i+1}/{num_sides}, go to j6 angle {goal_j1_angle}, trial: {total_trial_count}")
            franka_robot.tap_stick_y_joint(duration=record_duration/2, goal_j6_angle = goal_j1_angle)

        elif i == 3:
            print(f"tapping UP {i+1}/{num_sides}, go to j6 angle {goal_j1_angle * -1}, trial: {total_trial_count}")
            franka_robot.tap_stick_y_joint(duration=record_duration/2, goal_j6_angle = goal_j1_angle * -1)

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

        # detect if there has been a collision in the most recent audio recording
        collision_detected = mic_utils.detect_collision(save_path_data, total_trial_count)

        if not collision_detected:
            #delete the directory of the last trial
            directory_to_delete = save_path_data + f"trial{total_trial_count}/"
            os.system(f"rm -r {directory_to_delete}")
            # print(f"No collision! Deleted directory: {directory_to_delete}")

        else:
            print(f"collision detected!")
            
            # ************* make model prediction here *************
            if PREDICT_MODEL:
                height_pred, x_pred, y_pred = predict_contact_from_wavfile(cfg)
                height_i, x_i, y_i = height_pred[total_trial_count].item(), x_pred[total_trial_count].item(), y_pred[total_trial_count].item()
                print(f"trial: {total_trial_count}, height_pred: {height_i}, x_pred: {x_i}, y_pred: {y_i}")
                franka_robot.save_prediction( height_i, x_i,y_i , save_path_data, total_trial_count)
            
                #post process XY -> radian -> XY (to ensure projection is on the unit circle) AND (resolve wrap around issues) 
                cur_contact_pt = franka_robot.transform_predicted_XYZ_to_EE_XYZ(x_i, y_i,height_i, cfg.cylinder_radius, cfg.cylinder_transform_offset)
            
                time.sleep(2)
                create_marker_and_publish(franka_robot, pub_contactloc, cur_contact_pt, total_trial_count, pub_stick_markerarray, stick_markerarray)

                #transform predictions to global frame and then save
                franka_robot.save_prediction_global(total_trial_count, save_path_data) 

            # ****************************************************

            #at the end, increment the trial count
            total_trial_count += 1

        #restore robot to joint position
        time.sleep(1) #to give ample time to TF lookup before moving back
        print(f" ===== # moving back to precontact joints  ===== ")
        franka_robot.go_to_joint_position(joints_before_contact, duration=3)


def publish_tapping_pos_markers(tapping_pos, pub_tapping_pos):
    """
    Given the array of tapping positions, publish the markers in RVIZ once
    """

    #create markerarray for tapping positions
    tapping_markerarray = MarkerArray()

    for i in range(tapping_pos.shape[0]):
        # print(f"tapping_pos: {tapping_pos[i]}")
        tapping_point = tapping_pos[i]
        tapping_marker = Marker()
        tapping_marker.header.frame_id = "panda_link0"
        tapping_marker.header.stamp = rospy.Time.now()
        tapping_marker.ns = "pt"
        tapping_marker.id = i
        tapping_marker.type = Marker.SPHERE
        tapping_marker.action = Marker.ADD
        tapping_marker.pose.position.x = tapping_point[0]
        tapping_marker.pose.position.y = tapping_point[1]
        tapping_marker.pose.position.z = tapping_point[2]
        tapping_marker.pose.orientation.x = 0.0
        tapping_marker.pose.orientation.y = 0.0
        tapping_marker.pose.orientation.z = 0.0
        tapping_marker.pose.orientation.w = 1.0
        tapping_marker.scale.x = 0.01
        tapping_marker.scale.y = 0.01
        tapping_marker.scale.z = 0.01
        tapping_marker.color.a = 0.5
        tapping_marker.color.r = 0.0
        tapping_marker.color.g = 0.0
        tapping_marker.color.b = 1.0
        tapping_marker.lifetime = rospy.Duration()
        tapping_markerarray.markers.append(tapping_marker)

    pub_tapping_pos.publish(tapping_markerarray)

@hydra.main(version_base='1.3',config_path='../../learning/configs', config_name = 'inference')
def main(cfg: DictConfig):

    print(f" ------ starting script ------  ")

    #load pre-defined tapping positions
    tapping_pos_file_path = '/home/iam-lab/audio_localization/vibrotactile_localization/data/test_mapping/'
    tapping_pos_file_name = 'sampled_points.npy'
    tapping_pos = np.load(tapping_pos_file_path + tapping_pos_file_name)
    num_tapping_points = tapping_pos.shape[0]
    print(f"tapping_pos shape: {tapping_pos.shape}")

    #init franka robot
    franka_robot = FrankaMotion()

    # ROS publisher for the contact location
    pub_contactloc = rospy.Publisher('/contact_location', Marker, queue_size=10) #--> publish X: rad_X, Y: rad_Y, Z:height
    pub_stick_markerarray = rospy.Publisher('/stick_location', MarkerArray, queue_size=10)
    # ROS Publisher for the tapping positions
    pub_tapping_pos = rospy.Publisher('/tapping_positions', MarkerArray, queue_size=10)


    #markerarray for the rod
    stick_markerarray = MarkerArray()

    print(f" ===== #1. go to initial pose =====")
    if ROBOT_MOTION: franka_robot.go_to_init_pose()

    #create markerarray for tapping positions
    publish_tapping_pos_markers(tapping_pos, pub_tapping_pos)

    print(f" ===== #2. go to initial recording pose =====")
    if ROBOT_MOTION: franka_robot.go_to_init_recording_pose()

    robot_joints_restore_position = franka_robot.get_joints()

    print(f" ===== #3. rotate to j7 = 1.5708 (-90 deg) =====")
    if ROBOT_MOTION: franka_robot.rotate_j7(j7_radian_init, duration=5)

    #get ee rotm for tapping
    current_ee_RigidTransform_rotm = franka_robot.get_ee_pose().rotation

    #get inital recording x,y position
    initial_recording_pose = franka_robot.get_ee_pose()
    init_y, init_z = initial_recording_pose.translation[1], initial_recording_pose.translation[2]


    init_z_offset = init_z - cylinder_length/2



    #store information about the cylinder
    #get inital recording x,y position
    initial_recording_pose = franka_robot.get_ee_pose()
    init_y, init_z = initial_recording_pose.translation[1], initial_recording_pose.translation[2]

    for i in range(num_tapping_points):
        print(f" ===== tapping point {i+1}/{num_tapping_points} =====")

        #move franka to tapping position
        print(f" ===== #4. moving to tapping position =====")
        tapping_point = tapping_pos[i]
        print(f"tapping_point: {tapping_point}, go to pos {tapping_point[0]}, {tapping_point[1]}, {init_z}")
        if ROBOT_MOTION: 
            franka_robot.go_to_tapping_pose(tapping_point[0], tapping_point[1], init_z_offset, current_ee_RigidTransform_rotm)
            tap_all_four_directions(cfg,franka_robot, goal_j1_angle, pub_contactloc, pub_stick_markerarray, stick_markerarray)





    print(f" ------ ending script ------  ")

if __name__ == '__main__':
    main()