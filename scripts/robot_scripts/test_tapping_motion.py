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

# ================================================================================================================
ROBOT_MOTION = True
PREDICT_MODEL = True
data_recording = True # used when running eval for online inference only. TURN ON for offline data collection.
save_path_data = '/home/iam-lab/audio_localization/vibrotactile_localization/data/test_mapping/cross_easy_fullv4/'



devicelist=[2]
number_of_mics = 6
channels_in = 6
fs = 44100


number_of_hitting_samples = 4

cylinder_length = 0.203 #meters along the cylinder to traverse
goal_j1_angle = 17
record_duration = 2
total_trial_count = 0
# num_samples = 4
radian_sample_count = 4
j7_radian_min, j7_radian_max = -2.7, 2.7 #0,0 #-2.7, 2.7 #radians for tap stick joint
radian_along_cylinder = np.linspace(j7_radian_min, j7_radian_max, radian_sample_count)

output_prediction_list = []
output_avg_prediction_list = []

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

    elif stick_axis == 'x':
        x_stick_pos = np.linspace(rod_start_position[0], rod_end_position[0], number_of_hitting_samples)
        y_stick_pos = np.linspace(rod_start_position[1], rod_end_position[1], number_of_hitting_samples)
        z_stick_pos = np.linspace(rod_start_position[2], rod_end_position[2], number_of_hitting_samples)

        x_hitting_pos = x_stick_pos #varies
        y_hitting_pos = y_stick_pos - stick_thickness/2 - stick_tapping_offset # no change
        z_hitting_pos = z_stick_pos # no change

        x_hitting_pos_opposite = x_stick_pos
        y_hitting_pos_opposite = y_stick_pos + stick_thickness/2 + stick_tapping_offset
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


    # print(f"height_pred: {height_pred}, deg_pred: {deg_pred},  degree_diff: {degree_diff}")

    return height_pred, x_pred, y_pred

def create_marker_and_publish(franka_robot, pub_contactloc, contact_pt, total_trial_count, pub_stick_markerarray, stick_markerarray):
    scale_xyz_list = [0.05, 0.05, 0.05]
    color_argba_list = [1, 1, 0, 0]
    pred_marker = franka_robot.create_marker(total_trial_count, contact_pt,  scale_xyz_list, color_argba_list, lifetime=2, frame_id="cylinder_origin")
    
    #publish the predicted contact point
    pub_contactloc.publish(pred_marker)

    #create new marker for stick array
    array_scale_xyz_list = [0.03, 0.03, 0.03]
    array_color_argba_list = [1, 0, 1, 0]
    array_marker = franka_robot.create_marker(total_trial_count, contact_pt,  array_scale_xyz_list, array_color_argba_list, lifetime=30, frame_id="cylinder_origin")

    #publish the stick location
    stick_markerarray.markers.append(array_marker)
    pub_stick_markerarray.publish(stick_markerarray)

def create_marker_and_publish_avg_pts(franka_robot, cur_contact_pt, total_trial_count, pub_current_pt_markerarray, same_pts_markerarray):
    """
    create marker array to show 3 same points
    """

    #create new marker for stick array
    array_scale_xyz_list = [0.03, 0.03, 0.03]
    array_color_argba_list = [1, 0, 0, 1]
    array_marker = franka_robot.create_marker(total_trial_count, cur_contact_pt,  array_scale_xyz_list, array_color_argba_list, lifetime=30)

    # ------------ transform to global frame panda_link0 ----------------


    
    # -------------------------------------------------------------------
    #publish the stick location
    same_pts_markerarray.markers.append(array_marker)
    # pub_current_pt_markerarray.publish(same_pts_markerarray)



    
    


def average_contact_pts(contact_pts_to_average):
    """
    given a list of contact points, average them to get a single contact point
    """

    num_pts = len(contact_pts_to_average)
    x_avg, y_avg, z_avg = 0, 0, 0

    for pt in contact_pts_to_average:
        print(f"pt: {pt}")
        x_avg += pt.x
        y_avg += pt.y
        z_avg += pt.z

    x_avg = x_avg/num_pts
    y_avg = y_avg/num_pts
    z_avg = z_avg/num_pts


    avg_pt = Point(x=x_avg, y=y_avg, z=z_avg)
    print(f"avg_pt: {avg_pt}")
    return avg_pt



def robot_hit_along_list(hitting_location_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray, stick_markerarray, current_ee_RigidTransform_rotm, opposite_side=False, vertical_hit = False):
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
        if ROBOT_MOTION: franka_robot.move_with_fixed_orientation(x=hit_location[0], y=hit_location[1], z=hit_location[2], current_ee_RigidTransform_rotm = current_ee_RigidTransform_rotm)

        #get current j7 joint position
        j7_joint_radian = franka_robot.get_joints()[6]
        print(f"j7_joint_radian: {j7_joint_radian}") #--> -2.1

        #create radian_samples by adding and subtracting 0.6 radians from the current j7 joint position
        # radian_samples = [j7_joint_radian - 0.6, j7_joint_radian, j7_joint_radian + 0.6]
        radian_samples = [j7_joint_radian]

        #contact pt list to average and then publish to reduce noise
        contact_pts_to_average = []
        same_pts_markerarray = MarkerArray()

        #repeat hits at various radians to get a better estimate of the contact point
        for radians in radian_samples:

            #rotate j7 to the desired radian
            print(f" ===== #aab. rotate to j7 = {radians} =====")
            if ROBOT_MOTION: franka_robot.rotate_j7(radians, duration=5)

            #store current joint position
            joints_before_contact = franka_robot.get_joints()

            if opposite_side:
                hit_j1_angle = -1 * goal_j1_angle
            else:
                hit_j1_angle = goal_j1_angle

            #tap stick
            print(f" ===== #b. tapping stick joint at {hit_j1_angle} ===== ")
            if ROBOT_MOTION: 
                if vertical_hit: franka_robot.tap_stick_y_joint(duration=record_duration/2, goal_j6_angle = hit_j1_angle)
                else: franka_robot.tap_stick_joint(duration=record_duration/2, goal_j1_angle = hit_j1_angle)

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
                franka_robot.save_prediction( height_i, x_i,y_i , save_path_data, total_trial_count)
            # ****************************************************

            #post process XY -> radian -> XY (to ensure projection is on the unit circle) AND (resolve wrap around issues) 
            cur_contact_pt = franka_robot.transform_predicted_XYZ_to_EE_XYZ(x_i, y_i,height_i, cfg.cylinder_radius, cfg.cylinder_transform_offset)
            
            # contact_pts_to_average.append(cur_contact_pt)
            #publish makers (pub turned off for now!!!)
            # create_marker_and_publish_avg_pts(franka_robot, cur_contact_pt, total_trial_count, pub_current_pt_markerarray, same_pts_markerarray)


            # if total_trial_count != 0 and (total_trial_count+1) % len(radian_samples) == 0:
            #     #average the 3 contact points from 1 location into a single prediction
            #     contact_pt, transformed_points = franka_robot.average_markerarray(same_pts_markerarray)
            #     print(f"contact_pt: {contact_pt}")
            #     output_prediction_list.append(transformed_points)
            #     output_avg_prediction_list.append(contact_pt)
            #     print(f" ********************************")

            #     #publish makers
            #     # create_marker_and_publish(franka_robot, pub_contactloc, contact_pt, total_trial_count, pub_stick_markerarray, stick_markerarray)
  
            create_marker_and_publish(franka_robot, pub_contactloc, cur_contact_pt, total_trial_count, pub_stick_markerarray, stick_markerarray)

            #transform predictions to global frame and then save
            franka_robot.save_prediction_global(total_trial_count, save_path_data) 

            #restore robot to joint position
            print(f" ===== #c. moving back to precontact joints  ===== ")
            if ROBOT_MOTION: franka_robot.go_to_joint_position(joints_before_contact, duration=5)

            #increment trial count
            total_trial_count += 1

            #saving the output prediction list using save_path_data
            np.save(save_path_data + 'output_prediction_list.npy', output_prediction_list)
            np.save(save_path_data + 'output_avg_prediction_list.npy', output_avg_prediction_list)
        

        

def hit_along_both_sides(hitting_location_list, hitting_location_opposite_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray, stick_markerarray, initial_recording_pose):
    """
    motion to execute tapping motion along both sides of the stick (using horizontal motion - j1 motion)
    """
    print(f" ===== #4. Hit along {hitting_location_list} =====")
    robot_hit_along_list(hitting_location_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray, stick_markerarray, current_ee_RigidTransform_rotm = initial_recording_pose.rotation, opposite_side=False, vertical_hit = False)


    print(f" ===== #5. moving to opposite side of stick =====")

    if ROBOT_MOTION: franka_robot.move_delta_position(x=0, y = 0, z=0.20, duration=10)
    if ROBOT_MOTION: franka_robot.move_delta_position(x=-0.25, y = 0, z=0, duration=10)


    print(f" ===== #6. Hit along {hitting_location_opposite_list} =====")
    robot_hit_along_list(hitting_location_opposite_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray, stick_markerarray, current_ee_RigidTransform_rotm = initial_recording_pose.rotation, opposite_side=True, vertical_hit = False)

    #move to opposite side of the stick
    print(f" ===== #7. moving to opposite side of stick =====")

    if ROBOT_MOTION: franka_robot.move_delta_position(x=0, y = 0, z=0.20, duration=10)
    if ROBOT_MOTION: franka_robot.move_delta_position(x=0.25, y = 0, z=0, duration=10)

def hit_along_both_sides_vertical(hitting_location_list, hitting_location_opposite_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray, stick_markerarray, initial_recording_pose):
    """
    motion to execute tapping motion along both sides of the stick (using vertical motion - j6 motion)
    """
    print(f" ===== #4. Hit along {hitting_location_list} =====")
    robot_hit_along_list(hitting_location_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray, stick_markerarray, current_ee_RigidTransform_rotm = initial_recording_pose.rotation, opposite_side=False, vertical_hit = True)


    print(f" ===== #5. moving to opposite side of stick =====")

    if ROBOT_MOTION: franka_robot.move_delta_position(x=0, y = 0, z=0.20, duration=10)
    if ROBOT_MOTION: franka_robot.move_delta_position(x=0, y = 0.2, z=0, duration=10)


    print(f" ===== #6. Hit along {hitting_location_opposite_list} =====")
    robot_hit_along_list(hitting_location_opposite_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray, stick_markerarray, current_ee_RigidTransform_rotm = initial_recording_pose.rotation, opposite_side=True, vertical_hit = True)

    #move to opposite side of the stick
    print(f" ===== #7. moving to opposite side of stick =====")

    if ROBOT_MOTION: franka_robot.move_delta_position(x=0, y = 0, z=0.20, duration=10)
    if ROBOT_MOTION: franka_robot.move_delta_position(x=0, y = -0.2, z=0, duration=10)

def hit_along_one_side_only(hitting_location_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray, stick_markerarray, initial_recording_pose):
    """
    motion to execute tapping motion along one side of the stick
    """
    print(f" ===== #4. Hit along {hitting_location_list} =====")
    robot_hit_along_list(hitting_location_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, stick_markerarray, current_ee_RigidTransform_rotm = initial_recording_pose.rotation, opposite_side=False)

def motion_for_hitting_cross():
    """
    demo motion - cross easy demo.
    2 horizontal hitting motions
    2 vertical hitting motions
    """

def motion_for_hitting_stick(franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, initial_recording_pose):
    """
    demo motion - stick hitting demo
    2 horizontal hitting motion
    """
    x1,y1,z1 = 0, 0.27, 0.47
    x2,y2,z2 = 0, 0.33, 0.47

    rod_start_position = [x1,y1,z1]
    rod_end_position = [x2,y2,z2]
    stick_length = 0.15
    stick_thickness = 0.025
    stick_tapping_offset = 0.1
    stick_axis = 'y'

    hitting_location_list, hitting_location_opposite_list = get_hitting_location(rod_start_position, rod_end_position, number_of_hitting_samples, stick_length, stick_thickness, stick_tapping_offset, stick_axis)

    print(f"hit location list: {hitting_location_list}")
    
    #markerarray for the rod
    stick_markerarray = MarkerArray()

    # ---------------------------------- execute tapping motion along 1 length, both sides of stick ----------------------------------
    hit_along_both_sides(hitting_location_list, hitting_location_opposite_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, stick_markerarray, initial_recording_pose)
    # -------------------------------------------------------------------------------------------------------------------------------

# def test_tf_transform(cfg, franka_robot,pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray ):
#     """
#     debugging test for local prediction transform to global coordinate
#     """

#     total_trial_count = 0
#     stick_markerarray = MarkerArray()

#     # ****************************************************
#     height_pred, x_pred, y_pred = predict_contact_from_wavfile(cfg)
#     height_i, x_i, y_i = height_pred[total_trial_count].item(), x_pred[total_trial_count].item(), y_pred[total_trial_count].item()
#     print(f"trial: {total_trial_count}, height_pred: {height_i}, x_pred: {x_i}, y_pred: {y_i}")
#     # franka_robot.save_prediction( height_i, x_i,y_i , save_path_data, total_trial_count)
#     # ****************************************************

#     #post process XY -> radian -> XY (to ensure projection is on the unit circle) AND (resolve wrap around issues) 
#     cur_contact_pt = franka_robot.transform_predicted_XYZ_to_EE_XYZ(x_i, y_i,height_i, cfg.cylinder_radius, cfg.cylinder_transform_offset)
    
#     create_marker_and_publish(franka_robot, pub_contactloc, cur_contact_pt, total_trial_count, pub_stick_markerarray, stick_markerarray)

#     #TF lookup for marker to global frame (marker "0" to "panda_link0")
#     print(f"TF lookup for marker to global frame")
#     listener = tf.TransformListener()

#     # Wait for the frame to be available
#     target_frame = "panda_link0"
#     #convert total_trial_count to string
#     marker_id_frame = "marker_" + str(total_trial_count) 

#     listener.waitForTransform(target_frame, marker_id_frame, rospy.Time(0), rospy.Duration(4.0))
#     # Get the transformation
#     (trans, rot) = listener.lookupTransform(target_frame, marker_id_frame, rospy.Time(0))

#     print(f"transformation from {marker_id_frame} to {target_frame}. trans: {trans} ")

#     print(f"transform complete")


@hydra.main(version_base='1.3',config_path='../../learning/configs', config_name = 'inference')
def main(cfg: DictConfig):
    global total_trial_count

    print(f" ------ starting script ------  ")

    #init franka robot
    franka_robot = FrankaMotion()

    # ROS publisher for the contact location
    pub_contactloc = rospy.Publisher('/contact_location', Marker, queue_size=10) #--> publish X: rad_X, Y: rad_Y, Z:height
    pub_stick_markerarray = rospy.Publisher('/stick_location', MarkerArray, queue_size=10)
    pub_current_pt_markerarray = rospy.Publisher('/current_locations', MarkerArray, queue_size=10)



    print(f" ===== #1. go to initial pose =====")
    if ROBOT_MOTION: franka_robot.go_to_init_pose()
    print(f" ===== #2. go to initial recording pose =====")
    if ROBOT_MOTION: franka_robot.go_to_init_recording_pose()

    robot_joints_restore_position = franka_robot.get_joints()

    print(f" ===== #3. rotate to j7 = 1.5708 (-90 deg) =====")
    if ROBOT_MOTION: franka_robot.rotate_j7(-2.1, duration=5)

    #store information about the cylinder
    #get inital recording x,y position
    initial_recording_pose = franka_robot.get_ee_pose()
    init_y, init_z = initial_recording_pose.translation[1], initial_recording_pose.translation[2]
    j7_joint_radian = franka_robot.get_joints()[6]

    gt_label = [-0.101, j7_joint_radian] #[-0.101 fixed height, 0 fixed radian]
    print(f"gt label: {gt_label}")
    
    #horizontal, near side of cross
    x1,y1,z1 = 0, 0.24, 0.47
    x2,y2,z2 = 0, 0.33, 0.47

    #horizontal, far side of cross
    x3,y3,z3 = 0, 0.48, 0.47
    x4,y4,z4 = 0, 0.51, 0.47

    rod_start_position = [x1,y1,z1]
    rod_end_position = [x2,y2,z2]
    stick_length = 0.15
    stick_thickness = 0.025
    stick_tapping_offset = 0.1
    

    rod_start_position2 = [x3,y3,z3]
    rod_end_position2 = [x4,y4,z4]

    hitting_location_list, hitting_location_opposite_list = get_hitting_location(rod_start_position, rod_end_position, number_of_hitting_samples, stick_length, stick_thickness, stick_tapping_offset, stick_axis = 'y')
    hitting_location_list2, hitting_location_opposite_list2 = get_hitting_location(rod_start_position2, rod_end_position2, number_of_hitting_samples, stick_length, stick_thickness, stick_tapping_offset, stick_axis = 'y')

    print(f"hit location list: {hitting_location_list}")
    print(f"hitting location list2: {hitting_location_list2}")

    #vertical, left side of cross 
    x5,y5,z5 = 0.15, 0.42, 0.47
    x6,y6,z6 = 0.08, 0.42, 0.47

    rod_start_position3 = [x5,y5,z5]
    rod_end_position3 = [x6,y6,z6]

    #vertical, right side of cross 
    x7,y7,z7 = -0.15, 0.42, 0.47
    x8,y8,z8 = -0.08, 0.42, 0.47

    rod_start_position4 = [x7,y7,z7]
    rod_end_position4 = [x8,y8,z8]

    
    hitting_location_list3, hitting_location_opposite_list3 = get_hitting_location(rod_start_position3, rod_end_position3, number_of_hitting_samples, stick_length, stick_thickness, stick_tapping_offset, stick_axis = 'x')
    hitting_location_list4, hitting_location_opposite_list4 = get_hitting_location(rod_start_position4, rod_end_position4, number_of_hitting_samples, stick_length, stick_thickness, stick_tapping_offset, stick_axis = 'x')
    print(f"hit location list3: {hitting_location_list3}")
    print(f"hitting location list4: {hitting_location_list4}")

    
    #markerarray for the rod
    stick_markerarray = MarkerArray()

    # ---------------------------------- execute tapping motion along 1 length, both sides of stick ----------------------------------
    print(f" @@@ hit_along_both_sides_horizontal-near @@@ ")
    hit_along_both_sides(hitting_location_list, hitting_location_opposite_list, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray, stick_markerarray, initial_recording_pose)
    # -------------------------------------------------------------------------------------------------------------------------------

    print(f" @@@ hit_along_both_sides_vertical-Left @@@ ")
    hit_along_both_sides_vertical(hitting_location_list3, hitting_location_opposite_list3, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray, stick_markerarray, initial_recording_pose)
    
    print(f" @@@ hit_along_both_sides_vertical-Right @@@ ")
    if ROBOT_MOTION: franka_robot.move_delta_position(x=-0.35, duration=10)
    hit_along_both_sides_vertical(hitting_location_list4, hitting_location_opposite_list4, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray, stick_markerarray, initial_recording_pose)

    print(f" ------ moving +Z to next hitting location ------  ")
    #move robot EE up +z by 0.1, and then +y by 0.1
    if ROBOT_MOTION: franka_robot.move_delta_position(z=0.05, duration=10)
    if ROBOT_MOTION: franka_robot.move_delta_position(y=0.20, duration=10)
    if ROBOT_MOTION: franka_robot.move_delta_position(x=0.15, duration=10)

    print(f" ------ moving to next hitting location ------  ")
    # hit_along_one_side_only(hitting_location_list2, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, stick_markerarray, initial_recording_pose)
    hit_along_both_sides(hitting_location_list2, hitting_location_opposite_list2, franka_robot, gt_label, cfg, pub_contactloc, pub_stick_markerarray, pub_current_pt_markerarray, stick_markerarray, initial_recording_pose)

    #move robot EE up +z by 0.3
    print(f" ------ moving +Z before going to reset joint pos ------  ")
    if ROBOT_MOTION: franka_robot.move_delta_position(z=0.05, duration=10)

    
    #restore robot to home position
    print(f"restoring robot to home joints")
    if ROBOT_MOTION: franka_robot.reset_joints()

    
            
    print(f" ------ ending script ------  ")

if __name__ == '__main__':
    main()