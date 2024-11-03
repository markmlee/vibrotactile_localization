import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import pandas as pd
import sys
import imageio

import copy
import os
import time
import numpy as np
import sys
import torch
from tqdm import tqdm
import math
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

#urdfpy
from urdfpy import URDF
import numpy as np
import argparse

#ros
# import rospy
# import tf
from geometry_msgs.msg import Point, PoseStamped

#hydra
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig

#custom library for collision check
from eval_collisioncheck_calibration import Checker_Collision

#custom models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../learning')))
from models.CNN import CNNRegressor2D
from models.Resnet import ResNet18_audio, ResNet50_audio, ResNet50_audio_proprioceptive, ResNet50_audio_proprioceptive_dropout
from models.AudioSpectrogramTransformer import AST, AST_multimodal
from models.multimodal_transformer import MultiModalTransformer
from models.multimodal_transformer_xt_xdot import MultiModalTransformer_xt_xdot_t
from models.multimodal_transformer_xt_xdot_gcc import MultiModalTransformer_xt_xdot_t_gccphat
from models.multimodal_transformer_xt_xdot_gcc_tokens import MultiModalTransformer_xt_xdot_t_gccphat_tokens
from models.multimodal_transformer_xt_xdot_phase import MultiModalTransformer_xt_xdot_t_phase
from models.multimodal_transformer_xt_xdot_toda import MultiModalTransformer_xt_xdot_t_toda

#dataset
from datasets import AudioDataset
from eval_utils_plot import predict_from_eval_dataset

#custom utils
import microphone_utils as mic_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../learning')))
from transforms import to_mel_spectrogram, get_signal
import eval_utils as eval_utils

"""
The purpose of this script is to load the model and predict the output of the model on the given dataset
The dataset is a directory containing the audio files and joint state files

The output should match the predicted referrence first locally in the file predicted_output_HeightRad.npy
The output should match the ground truth reference second locally in the file gt_label.npy
The output should match the predicted referrence third globally in the file predicted_pt.npy

"""

CREATE_GIF_ROBOT_VISUALIZATION = False #set to True to create GIF of all trials of robot hitting the cylinder with contact
CREATE_GIF_ALL_CONTACTS = True #set to True to create GIF of all predictions, camera pannign viewpoints
SAVE_COLLISION_IMAGES = False #set to True to save images of collision check

def xy_to_radians( x, y):
        """
        Convert x,y into radians from 0 to 2pi
        """
        rad = np.arctan2(y, x)
        if rad < 0:
            rad += 2*np.pi

        return rad

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
    #TODO: MODIFY ACCORDING TO MODEL
    # model = CNNRegressor2D(cfg)
    # model = ResNet50_audio(cfg)
    # model = AST(cfg)
    # model = ResNet50_audio_proprioceptive(cfg)
    # model = ResNet50_audio_proprioceptive_dropout(cfg)
    # model = AST_multimodal(cfg)
    # model = MultiModalTransformer(cfg)
    # model = MultiModalTransformer_xt_xdot_t(cfg)
    model = MultiModalTransformer_xt_xdot_t_gccphat(cfg)
    # model = MultiModalTransformer_xt_xdot_t_phase(cfg)
    # model = MultiModalTransformer_xt_xdot_t_toda(cfg)
    # model = MultiModalTransformer_xt_xdot_t_gccphat_tokens(cfg)
    model.load_state_dict(torch.load(os.path.join(cfg.model_directory, 'model.pth')))

    #verify if model is loaded by checking the model parameters
    # print(model)
    model.to(device)
    model.eval()

    #print length of dataset
    total_trials = len(dataset)
    print(f"total_trials: {total_trials}")

    # predict
    height_pred, x_pred, y_pred, _ , _, _, _, _, _, _ =  predict_from_eval_dataset(cfg, model, device, val_loader)

    return height_pred, x_pred, y_pred, total_trials

def compare_offline_prediction_with_saved_predictions(cfg, height_pred, x_pred, y_pred, total_trials):
    """
    Iterate through directory and then compare the predicted output with the saved output
    predicted output: passed in params
    saved output: loaded from cfg directory / trial / predicted_output_HeightRad.npy
    Saved output is named as predicted_output_HeightRad.npy
    Return the MAE error in heightx,y
    """
    
    MAE_height = 0
    MAE_deg = 0
    MAE_gt_height = 0
    MAE_gt_deg = 0

    for i in range(total_trials):
        #load the saved output using cfg.data_dir,"trial{i}", "predicted_output_HeightRad.npy"
        saved_output = np.load(os.path.join(cfg.data_dir, f"trial{i}", "predicted_output_HeightRad.npy"))
        
        # print(f"saved_output: {saved_output}") #height, radian
        height_saved = saved_output[0]
        rad_saved = saved_output[1]
        
        # print(f"saved height: {height_saved}, saved rad: {rad_saved}")

        #convert XY into rad for prediction

        rad_pred = xy_to_radians(y_pred[i], x_pred[i])

        # print(f"rad_pred: {rad_pred}, rad_saved: {rad_saved}")

        #calculate the error
        height_error = np.abs(height_pred[i] - height_saved)
        radian_error = np.abs(rad_pred - rad_saved)

        #rad to deg
        deg_error = np.rad2deg(radian_error)
        
        # print(f"pred height: {height_pred[i]}, pred rad: {rad_pred}")
        # print(f"height_error: {height_error}, radian_error: {radian_error}, deg_error: {deg_error}")


        #load the GT output using cfg.data_dir,"trial{i}", "gt_label.npy"
        gt_output = np.load(os.path.join(cfg.data_dir, f"trial{i}", "gt_label.npy"))
        gt_height = gt_output[0]
        gt_rad = gt_output[1]

        if gt_rad < 0:
            gt_rad += 2*np.pi

        #calculate the error between gt and pred
        gt_height_error = np.abs(height_pred[i] - gt_height)
        gt_radian_error = np.abs(rad_pred - gt_rad)

        #rad to deg
        gt_deg_error = np.rad2deg(gt_radian_error)

        # print(f"gt height: {gt_height}, gt rad: {gt_rad}, pred height: {height_pred[i]}, pred rad: {rad_pred}")
        # print(f"gt height_error: {gt_height_error}, gt deg_error: {gt_deg_error}")

        MAE_height += height_error
        MAE_deg += deg_error
        MAE_gt_height += gt_height_error
        MAE_gt_deg += gt_deg_error

    MAE_height = MAE_height / total_trials
    MAE_deg = MAE_deg / total_trials
    MAE_gt_height = MAE_gt_height / total_trials
    MAE_gt_deg = MAE_gt_deg / total_trials

    return MAE_height, MAE_deg, MAE_gt_height, MAE_gt_deg


def radians_to_xy_on_cylinder(rad, cylinder_radius):
    """
    Convert radians to x,y with radius included
    """
    x = np.cos(rad) * cylinder_radius
    y = np.sin(rad) * cylinder_radius

    return x, y

def transform_origin_to_cylinder(rad_input, cylinder_transform_offset):
    """
    Transform the arbitraty contact pt origin during data collection to the actual cylinder EE origin.
    Abritrary origin during dataset collection - j7 measurement at 0 deg is  approx 45 deg offset to contact.
    Return the transformed radians by subrtacting 45 deg offset .
    """

    # print(f"rad_input: {rad_input}, degrees: {np.degrees(rad_input)}")

    rad = -1*rad_input - np.radians(cylinder_transform_offset)

    rad2 = rad_input + np.radians(cylinder_transform_offset)
    # print(f"rad: {rad}, rad2: {rad2}")
    # if rad < 0:
    #     rad += 2*np.pi

    return rad

def transform_predicted_XYZ_to_EE_XYZ(x,y,z, cylinder_radius, cylinder_transform_offset):
    """
    Transform the predicted contact pt XYZ (based on dataset cylinder frame) to the EE XYZ (to visualize on RVIZ on EE frame)
    1. post process XY to radian back to XY
    2. align the origin of the cylinder frame to the EE origin
    3. convert to Point Msg
    """

    #convert xy into radians, then project back to x,y with radius mult
    radians = xy_to_radians(x, y)

    #transform origin to cylinder EE origin
    radians = transform_origin_to_cylinder(radians, cylinder_transform_offset)
    x_on_cylinder, y_on_cylinder = radians_to_xy_on_cylinder(radians, cylinder_radius)


    transformed_point = Point()
    transformed_point.x = x_on_cylinder
    transformed_point.y = y_on_cylinder
    transformed_point.z = (-1*z / 100)  #origin opposite from dataset and RViz. convert cm to m
    transformed_point.z += 0.0 #add fine tuning offset

    return transformed_point
    
def load_robot_and_cylinder(urdf_dir):
    """
    Load the robot urdf file and return the robot object
    Load the cylinder end-effector from the robot pose
    """

    robot = URDF.load(urdf_dir + 'panda.urdf')

    cylinder = o3d.io.read_triangle_mesh(urdf_dir + '/meshes/cylinder/EE_cylinder_dense.stl')
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color([0.0, 0.0, 1.0])
    #paint x>0.0 vertices of cylinder green
    colors = np.asarray(cylinder.vertex_colors)
    verts = np.asarray(cylinder.vertices)
    idx = np.where(verts[:,0]>0.0)[0]
    colors[idx] = [0.0, 1.0, 0.0]
    cylinder.vertex_colors = o3d.utility.Vector3dVector(colors)

    return robot, cylinder



def get_arm_pose_from_robotstate(trial_dir, robot):
    """
    Load the robot state from the trial directory, do FK and return the pose of the end-effector
    Returns a 4x4 Trans matrix
    """

    joints = np.load(trial_dir + '/q_t.npy')
    pose= robot.link_fk(links=['panda_hand'], cfg={
                                'panda_joint1':joints[-125,0],
                                'panda_joint2':joints[-125,1],
                                'panda_joint3':joints[-125,2],
                                'panda_joint4':joints[-125,3],
                                'panda_joint5':joints[-125,4],
                                'panda_joint6':joints[-125,5],
                                'panda_joint7':joints[-125,6]})
    #pose= pose['panda_hand']
    # print("keys", pose.keys())
    for key in pose.keys():
        pose = pose[key]
    # print("pose:", pose)

    joint_trajectory = joints[-125-30:-125,:] #last 30 points

    joints = joints[-125,:]
    # Convert joints to a list of lists
    joints_list = joints.tolist()

    
    
    return pose, joints_list, joint_trajectory

def convert_contactpt_to_global(cur_contact_pt, robot, robot_joints):
    """
    Convert the contact point w.r.t to the /cylinder_origin frame to the /panda_joint1 frame

    cur_contact_pt is w.r.t to the /cylinder_origin frame, which needs to be created
    /cylinder_origin frame is translated from /panda_hand frame by [0,0,-0.155]
    Returned contact_pt_global is w.r.t to the /panda_joint1 frame
    Returned the /cylinder_origin w.r.t. to the /panda_joint1 frame
    """
    # Step 1: Get the transformation from panda_link1 to panda_hand
    pose_hand = robot.link_fk(links=['panda_hand'], cfg={
        'panda_joint1': robot_joints[0],
        'panda_joint2': robot_joints[1],
        'panda_joint3': robot_joints[2],
        'panda_joint4': robot_joints[3],
        'panda_joint5': robot_joints[4],
        'panda_joint6': robot_joints[5],
        'panda_joint7': robot_joints[6],
    })

    for key in pose_hand.keys():
        # print("key:", key)
        # print(f"pose_hand[key]: {pose_hand[key]}")
        pose_hand = pose_hand[key]
    # print("pose_hand:", pose_hand)

    # Step 2: Create the transformation from panda_hand to cylinder_origin
    T_hand_cylinder = np.eye(4)
    T_hand_cylinder[2, 3] = 0.155  # Translate by [0, 0, -0.155]

    # Step 3: Combine transformations
    T_link1_cylinder = pose_hand @ T_hand_cylinder

    # Step 4: Convert cur_contact_pt to homogeneous coordinates
    contact_pt_cylinder = np.array([cur_contact_pt.x, cur_contact_pt.y, cur_contact_pt.z, 1])

    # print(f"T_link1_cylinder : {T_link1_cylinder}")
    # print(f"contact_pt_cylinder: {contact_pt_cylinder}")
    

    # Step 5: Transform the point
    contact_pt_link1 = T_link1_cylinder @ contact_pt_cylinder

    # print(f"contact_pt_link1: {contact_pt_link1}")

    # Step 6: Convert back to Point message
    contact_pt_global = Point()
    # contact_pt_global.x = contact_pt_link1[0,-1]
    # contact_pt_global.y = contact_pt_link1[1,-1]
    # contact_pt_global.z = contact_pt_link1[2,-1]

    contact_pt_global.x = contact_pt_link1[0]
    contact_pt_global.y = contact_pt_link1[1]
    contact_pt_global.z = contact_pt_link1[2]


    # print(f"contact_pt_global: {contact_pt_global}")
    # print(f"x: {contact_pt_global.x}, y: {contact_pt_global.y}, z: {contact_pt_global.z}")

    # Step 7: Also create a Point message for the /cylinder_origin w.r.t. to the /panda_joint1 frame
    cylinder_origin_link1 = Point()
    cylinder_origin_link1.x = T_link1_cylinder[0, 3]
    cylinder_origin_link1.y = T_link1_cylinder[1, 3]
    cylinder_origin_link1.z = T_link1_cylinder[2, 3]

    
    return contact_pt_global, cylinder_origin_link1


def get_stick_cylinder(path_to_stl):
    """
    For loading the single stick cylinder mesh
    """
    # Load collision object mesh
    collision_obj = o3d.io.read_triangle_mesh(path_to_stl)
    collision_obj.compute_vertex_normals()

    # Initial quaternion
    initial_quat = R.from_quat([0.7071, 0, 0, 0.7071])

    # Quaternion for 90-degree rotation around Z axis
    z_rotation_quat = R.from_quat([0, 0, 0.7071, 0.7071]) #THIS IS FOR VERTICAL STICK MOUNTS
    # z_rotation_quat = R.from_quat([0, 0, 0, 1])


    # Compute the new quaternion by multiplying the initial quaternion by the Z rotation quaternion
    new_quat = z_rotation_quat * initial_quat

    r = R.from_quat(new_quat.as_quat())

    rz = R.from_euler('z', 2, degrees=True)
    T_object = np.eye(4)
    T_object[:3,:3] = rz.as_matrix()@r.as_matrix()
    T_object[0:3,3] = [-0.00,0.44,0.32]
    collision_obj.transform(T_object)

    return collision_obj


def manual_transformation_from_predetermined_transformation():
    """
    Have predetermined the transformation matrix from determine_transformation_from_scanned_obj()

    """

    transformation_final = np.array([
    [-0.7760436,   0.62971646,  0.03483545, -0.02671042],
    [-0.04891869, -0.00503359, -0.99879008,  0.70680278],
    [-0.62877921, -0.77680875,  0.03471119,  1.65692942],
    [ 0,           0,           0,           1]
])
    return transformation_final


def get_pcloud_cross(path_to_pts):

    # Load point cloud data
    df = pd.read_csv(path_to_pts, sep=r'\s+', header=None, comment='#', skiprows=1)

    # Convert to numpy array
    gt_pointcloud = df.values

    # Ensure the point cloud is of type float64 or float32
    gt_pointcloud = gt_pointcloud.astype(np.float64)  # or np.float32

    print(f"gt_pointcloud shape: {gt_pointcloud.shape}")

    # Create Open3D PointCloud object
    pcd_ground_truth = o3d.geometry.PointCloud()
    pcd_ground_truth.points = o3d.utility.Vector3dVector(gt_pointcloud[:, :3])

    # denoise the pointcloud
    cl, ind = pcd_ground_truth.remove_radius_outlier(nb_points=16, radius=0.009)

    inlier_cloud = pcd_ground_truth.select_by_index(ind)
    outlier_cloud = pcd_ground_truth.select_by_index(ind, invert=True)

    voxel_size = 0.005
    inlier_cloud = inlier_cloud.voxel_down_sample(voxel_size)

    # print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


    pcd_ground_truth = inlier_cloud


    # Get pre-determined transformation of scanned object to rough alignment with prediction coordinate system
    transformation_final =  manual_transformation_from_predetermined_transformation()

    # Create an additional translation matrix for 3cm in Z direction (manual translate)
    manual_translation = np.eye(4)
    manual_translation[1, 3] = 0.03  # 3cm down
    manual_translation[0, 3] = -0.01  # 2cm right 
    manual_translation[2, 3] = 0.0  # 2cm +z  

    # Combine the transformations
    combined_transformation = manual_translation @ transformation_final

    # Transform the ground truth point cloud to the prediction coordinate system
    pcd_ground_truth.transform(combined_transformation)

    return pcd_ground_truth

def create_prediction_visualization_gif(cur_contact_pt_list, cylinder_pose_list, gt_contact_point_list, num_frames=30, save_gif=True):
    """
    Create a GIF of the predicted contact points with varying viewpoints
    Input: List of predicted contact points and cylinder poses, and the ground truth contact pt 
    """

    # Define window dimensions
    window_width = 800
    window_height = 600

    # Create an Open3D visualization window with explicit size
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_width, height=window_height)
    
    # Add collision object to the visualizer
    add_collision_object_to_visualizer(vis)

    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # Add all collision points to the visualizer
    add_contact_pts_to_visualizer(vis, cur_contact_pt_list, cylinder_pose_list, gt_contact_point_list)

    #params for camerea to create GIF
    # Base camera parameters
    base_front = np.array([-0.75, 2, 0.5])
    up = [0.4, 0.15, 1.5]
    lookat = [0, 0, 0.5]
    zoom = 1.2


    if save_gif:
        # Create directory for temporary images
        if not os.path.exists(f"output/temp_images"):
            os.makedirs(f"output/temp_images")

        # Generate frames
        for i in range(num_frames):
 
            # Calculate rotation angle (if you want camera movement)
            angle = np.radians(180 * math.sin(2 * math.pi * i / num_frames))
            
            # Rotate front vector
            rotation = R.from_rotvec(angle * np.array([0, 0, 1]))
            rotated_front = rotation.apply(base_front)

            # Set camera view
            ctr = vis.get_view_control()
            ctr.set_front(rotated_front)
            ctr.set_up(up)
            ctr.set_lookat(lookat)
            ctr.set_zoom(zoom)

            # Render and capture image
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(f"output/temp_images/frame_{i:03d}.png")

        vis.destroy_window()

        # Create GIF
        with imageio.get_writer(f"output/predictions.gif", mode='I', duration=0.1) as writer:
            for i in range(num_frames):
                image = imageio.imread(f"output/temp_images/frame_{i:03d}.png")
                writer.append_data(image)

        print(f"GIF created: output/predictions.gif")

    else:
        # Display the visualizer
        vis.run()
        vis.destroy_window()


def create_robot_visualization_gif(trial_count, cylinders, cylinder_copy, robot, robot_joints, robot_joint_trajectory, contact_pt_transformed, num_frames=30, save_gif=False):
    """
    Create a GIF of the robot visualization with varying viewpoints and robot movement
    """
    # Define window dimensions
    window_width = 800
    window_height = 600

    # Create an Open3D visualization window with explicit size
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_width, height=window_height)
    

    # Add geometries to the visualizer
    add_geometries_to_visualizer(vis, cylinder_copy, contact_pt_transformed)

    # Add initial robot mesh
    robot_geometries = add_robot_to_visualizer(vis, robot, robot_joints)

    # Base camera parameters
    base_front = np.array([-0.75, 2, 0.5])
    up = [0.4, 0.15, 1.5]
    lookat = [0, 0, 0.5]
    zoom = 0.85

    if save_gif:
        # Create directory for temporary images
        if not os.path.exists(f"output/trial{trial_count}/temp_images"):
            os.makedirs(f"output/trial{trial_count}/temp_images")

        # Generate frames
        for i in range(num_frames):
            # Get the current joint configuration
            current_joints = robot_joint_trajectory[i]

            # Update robot position
            update_robot_position(vis, robot, current_joints, robot_geometries)

            # Calculate rotation angle (if you want camera movement)
            angle = np.radians(0 * math.sin(2 * math.pi * i / num_frames))
            
            # Rotate front vector
            rotation = R.from_rotvec(angle * np.array([0, 0, 1]))
            rotated_front = rotation.apply(base_front)

            # Set camera view
            ctr = vis.get_view_control()
            ctr.set_front(rotated_front)
            ctr.set_up(up)
            ctr.set_lookat(lookat)
            ctr.set_zoom(zoom)

            # Render and capture image
            vis.poll_events()
            vis.update_renderer()
            vis.capture_screen_image(f"output/trial{trial_count}/temp_images/frame_{i:03d}.png")

        vis.destroy_window()

        # Create GIF
        with imageio.get_writer(f"output/trial{trial_count}.gif", mode='I', duration=0.1) as writer:
            for i in range(num_frames):
                image = imageio.imread(f"output/trial{trial_count}/temp_images/frame_{i:03d}.png")
                writer.append_data(image)

        print(f"GIF created: output/trial{trial_count}.gif")

    else:
        # Set camera view
        ctr = vis.get_view_control()
        ctr.set_front(base_front)
        ctr.set_up(up)
        ctr.set_lookat(lookat)
        ctr.set_zoom(zoom)

        # Run the visualization
        vis.run()
        vis.destroy_window()

def update_robot_position(vis, robot, robot_joints, robot_geometries):
    """
    Update the position of the robot mesh
    """
    # Get the new visual geometries for the robot
    new_visual_geometries = robot.visual_trimesh_fk(cfg={
        f'panda_joint{i+1}': robot_joints[i] for i in range(7)
    })

    # Update each geometry
    for geom, (trimesh_obj, transform) in zip(robot_geometries, new_visual_geometries.items()):
        # Update vertices and triangles
        geom.vertices = o3d.utility.Vector3dVector(np.asarray(trimesh_obj.vertices))
        geom.triangles = o3d.utility.Vector3iVector(np.asarray(trimesh_obj.faces))
        
        # Apply the new transformation
        geom.transform(transform)
        
        # Compute vertex normals for proper shading
        geom.compute_vertex_normals()
        
        # Update the geometry in the visualizer
        vis.update_geometry(geom)


def add_collision_object_to_visualizer(vis):
    """
    Add the collision object to the visualizer
    Input: path to the collision object STL file
    """
    #TODO: choose which object to visualize
    # Add the collision object to the visualizer
    # collision_obj = get_stick_cylinder("/home/mark/audio_learning_project/acoustic_cylinder/franka_panda/meshes/cylinder/collision_object_dense.stl") #FOR STICK SIMPLE CASE


    gt_pointcloud_path = '/home/mark/audio_learning_project/evaluation/3D_scan_GT'
    gt_pointcloud_file = os.path.join(gt_pointcloud_path, 'OBJECT 1A cross.pts')
    collision_obj = get_pcloud_cross(gt_pointcloud_file)

    # Downsample the point cloud
    voxel_size = 0.01  # Adjust the voxel size as needed
    collision_obj_downsampled = collision_obj.voxel_down_sample(voxel_size)

    # Set the color of the point cloud to neon green
    colors = np.tile([0.0, 1.0, 0.0], (len(collision_obj_downsampled.points), 1))  # Neon green color
    collision_obj_downsampled.colors = o3d.utility.Vector3dVector(colors)

    vis.add_geometry(collision_obj_downsampled)


def add_contact_pts_to_visualizer(vis, cur_contact_pt_list, cylinder_pose_list, gt_contact_point_list):
    """
    Add all contact points (predicted and ground truth) to the visualizer, add all cylinder poses to the visualizer.
    Contact points are red spheres, cylinder poses are blue spheres.
    For each corresponding index of contact point and cylinder pose, add a line between them.
    """
    contact_spheres = []
    cylinder_spheres = []
    gt_contact_triangles = []
    lines = []

    for contact_pt, cylinder_pose, gt_contact_pt in zip(cur_contact_pt_list, cylinder_pose_list, gt_contact_point_list):

        # print(f"contact_pt: {contact_pt}, cylinder_pose: {cylinder_pose}")

        # Create a red sphere for the contact point
        contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        contact_sphere.paint_uniform_color([1.0, 0.0, 0.0])
        contact_sphere.translate(np.array([contact_pt.x, contact_pt.y, contact_pt.z]))
        vis.add_geometry(contact_sphere)
        contact_spheres.append(contact_sphere)

        # Create a blue sphere for the cylinder pose
        cylinder_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        cylinder_sphere.paint_uniform_color([0, 0, 1.0])
        cylinder_sphere.translate(np.array([cylinder_pose.x, cylinder_pose.y, cylinder_pose.z]))
        vis.add_geometry(cylinder_sphere)
        cylinder_spheres.append(cylinder_sphere)

        # Create a green triangle for the ground truth contact point
        triangle = o3d.geometry.TriangleMesh()
        vertices = np.array([
            [gt_contact_pt.x, gt_contact_pt.y, gt_contact_pt.z],
            [gt_contact_pt.x + 0.01, gt_contact_pt.y, gt_contact_pt.z],
            [gt_contact_pt.x, gt_contact_pt.y + 0.01, gt_contact_pt.z]
        ])
        triangles = np.array([[0, 1, 2]])
        triangle.vertices = o3d.utility.Vector3dVector(vertices)
        triangle.triangles = o3d.utility.Vector3iVector(triangles)
        triangle.paint_uniform_color([0.0, 0.0, 0.0])  # Green color
        vis.add_geometry(triangle)
        gt_contact_triangles.append(triangle)

        # Create a line between the contact point and the cylinder pose
        points_contact_to_cylinder = [
            [contact_pt.x, contact_pt.y, contact_pt.z],
            [cylinder_pose.x, cylinder_pose.y, cylinder_pose.z]
        ]
        lines.append([0, 1])
        line_set_contact_to_cylinder = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_contact_to_cylinder),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set_contact_to_cylinder.paint_uniform_color([0.0, 0.0, 0.0])  # Black line
        vis.add_geometry(line_set_contact_to_cylinder)

        # Create a line between the cylinder pose and the ground truth contact point
        points_cylinder_to_gt = [
            [cylinder_pose.x, cylinder_pose.y, cylinder_pose.z],
            [gt_contact_pt.x, gt_contact_pt.y, gt_contact_pt.z]
        ]
        lines.append([0, 1])
        line_set_cylinder_to_gt = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_cylinder_to_gt),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set_cylinder_to_gt.paint_uniform_color([0.0, 0.0, 0.0])  # Black line
        vis.add_geometry(line_set_cylinder_to_gt)

    return contact_spheres, cylinder_spheres, lines

def add_geometries_to_visualizer(vis, cylinder_copy, contact_pt_transformed):
    """
    Update the visualizer with the transformed cylinder, collision object, and contact point
    """
    # Add the transformed cylinder
    # vis.add_geometry(cylinder_copy)

    # Add the collision object (path is hardcoded for now)
    add_collision_object_to_visualizer(vis)

    # contact_pt = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    # contact_pt.paint_uniform_color([1.0, 0.0, 0.0])
    # contact_pt.translate(np.array([contact_pt_transformed.x, contact_pt_transformed.y, contact_pt_transformed.z]))
    # vis.add_geometry(contact_pt)

    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)


def add_robot_to_visualizer(vis, robot, robot_joints):
    """
    Add the robot's wireframe to the visualizer and return the geometries
    """
    robot_geometries = []

    # Get the visual geometries for the robot
    visual_geometries = robot.visual_trimesh_fk(cfg={
        f'panda_joint{i+1}': robot_joints[i] for i in range(7)
    })

    # Add the robot to the visualizer as wireframe
    for trimesh_obj, transform in visual_geometries.items():
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(trimesh_obj.vertices))
        o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(trimesh_obj.faces))
        o3d_mesh.transform(transform)
        
        # wire_frame = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
        # wire_frame.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray color

        # Compute vertex normals for proper shading
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray color
        vis.add_geometry(o3d_mesh)
        robot_geometries.append(o3d_mesh)

    return robot_geometries

def convert_list_of_points_to_pcloud(transformed_contact_pt_list):
    """
    Convert a list of contact points to an Open3D point cloud
    Input: List of contact points in Point message
    Output: Open3D point cloud
    """

    # Extract the coordinates from the list of Point messages
    points = np.array([[pt.x, pt.y, pt.z] for pt in transformed_contact_pt_list])

    # Create an Open3D PointCloud object
    pcloud_of_contactpts = o3d.geometry.PointCloud()
    pcloud_of_contactpts.points = o3d.utility.Vector3dVector(points)

    return pcloud_of_contactpts
 
@hydra.main(version_base='1.3',config_path='../learning/configs', config_name = 'inference')
def main(cfg: DictConfig):
    # -------------------------------------------------------
    print(f"**** start script **** ")
    
    #load the robot and cylinder
    urdf_dir = '/home/mark/audio_learning_project/acoustic_cylinder/franka_panda/'
    robot, cylinder = load_robot_and_cylinder(urdf_dir)

    # get GT point cloud
    gt_pointcloud_path = '/home/mark/audio_learning_project/evaluation/3D_scan_GT'
    gt_pointcloud_file = os.path.join(gt_pointcloud_path, 'OBJECT 1A cross.pts')
    collision_obj_pcloud = get_pcloud_cross(gt_pointcloud_file)

    #create dataset from the directory
    #load the model
    #predict the output
    height_pred, x_pred, y_pred, total_trials = predict_contact_from_wavfile(cfg)

    #convert tensor to np array
    height_pred = height_pred.cpu().numpy()
    x_pred = x_pred.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    print(f"size of height_pred: {height_pred.shape}, x_pred: {x_pred.shape}, y_pred: {y_pred.shape}")

    # MAE_bewteen_predictions_height, MAE_bewteen_predictions_deg, MAE_gt_height, MAE_gt_deg = compare_offline_prediction_with_saved_predictions(cfg, height_pred, x_pred, y_pred, total_trials)
    # print(f"MAE_height: {MAE_bewteen_predictions_height}, MAE_deg: {MAE_bewteen_predictions_deg}, MAE_gt_height: {MAE_gt_height}, MAE_gt_deg: {MAE_gt_deg}")
    
    transformed_contact_pt_list = [] #visualizing contacts
    cylinder_pose_list = [] #visualizing robot ee pose

    # get all robot_joints by iterating through the trial directories
    ee_pose_list = []
    robot_joints_list = []
    robot_joint_trajectory_list = []
    
    for trial_count in range(total_trials): 
        # load the robot state from the trial directory 
        trial_dir = cfg.data_dir + '/trial' + str(trial_count)
        ee_pose, robot_joints, robot_joint_trajectory = get_arm_pose_from_robotstate(trial_dir, robot)
        ee_pose_list.append(ee_pose)
        robot_joints_list.append(robot_joints)
        robot_joint_trajectory_list.append(robot_joint_trajectory)


    # create a collision checker instance
    # Initialize Checker_Collision
    initial_joints = robot_joints_list[0]
    collision_checker = Checker_Collision(cylinder, collision_obj_pcloud, robot, initial_joints)

    # Setup visualization
    collision_checker.setup_visualization(window_width=800, window_height=600)

    # Run collision check
    min_dist_list, image_list, gt_contact_point_list = collision_checker.run_collision_check(robot_joints_list, visualize=SAVE_COLLISION_IMAGES)

    if SAVE_COLLISION_IMAGES:
        # Save the images in the output directory with the current timestamp
        output_dir = "output_images"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir_with_timestamp = os.path.join(output_dir, timestamp)

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir_with_timestamp, exist_ok=True)

        for i, image in enumerate(image_list):
            image_path = os.path.join(output_dir_with_timestamp, f"image_{i}.png")
            imageio.imwrite(image_path, image)
            print(f"Saved image {i} to {image_path}")

        sys.exit()


    ###transform the output to robot baselink frame###
    #post process XY -> radian -> XY (to ensure projection is on the unit circle) AND (resolve wrap around issues) 
    for trial_count in range(total_trials):
        print(f"trial: {trial_count} ,input x: {x_pred[trial_count]}, y: {y_pred[trial_count]}, height: {height_pred[trial_count]}")
        cur_contact_pt = transform_predicted_XYZ_to_EE_XYZ(x_pred[trial_count], y_pred[trial_count],height_pred[trial_count], cfg.cylinder_radius, cfg.cylinder_transform_offset)
        # print(f"cur_contact_pt: {cur_contact_pt}")

        
        #transform the cylinder to the EE pose
        cylinder_copy = copy.deepcopy(cylinder)
        cylinder_copy.transform(ee_pose_list[trial_count])
        cylinder_verts = np.asarray(cylinder_copy.vertices)
        cylinder_colors  = np.asarray(cylinder_copy.vertex_colors)

        #transform the contact point to the global frame
        contact_pt_transformed, robot_ee_pose = convert_contactpt_to_global(cur_contact_pt, robot, robot_joints_list[trial_count])

        if CREATE_GIF_ROBOT_VISUALIZATION:
            create_robot_visualization_gif(trial_count, cylinder, cylinder_copy, robot, robot_joints_list[trial_count], robot_joint_trajectory_list[trial_count], contact_pt_transformed, save_gif=True)


        # print(f"contact_pt_transformed: {contact_pt_transformed}, robot_ee_pose: {robot_ee_pose}")
        transformed_contact_pt_list.append(contact_pt_transformed)
        cylinder_pose_list.append(robot_ee_pose)


    print(f" ******* comparing pred and GT ******* ")
    for trial_count in range(total_trials):
        gt_contact_pt = gt_contact_point_list[trial_count]
        transformed_contact_pt = transformed_contact_pt_list[trial_count]
        print(f"trial: {trial_count}, gt_contact_point xyz: ({gt_contact_pt.x:.4f}, {gt_contact_pt.y:.4f}, {gt_contact_pt.z:.4f}), "
            f"transformed_contact_pt xyz: ({transformed_contact_pt.x:.4f}, {transformed_contact_pt.y:.4f}, {transformed_contact_pt.z:.4f})")
    

    # # ================== Post processing ==================
    delete_trial_index = []

    for trial_count in range(total_trials):
        # print only 3 decimal places
        print(f"cylinder_pos_X: {cylinder_pose_list[trial_count].x:.3f}, "
            f"cylinder_pos_Y: {cylinder_pose_list[trial_count].y:.3f}, "
            f"cylinder_pos_Z: {cylinder_pose_list[trial_count].z:.3f}")
        
        # # if y value > 0.53, remove this trial from  cylinder_pose_list and transformed_contact_pt_list
        # if cylinder_pose_list[trial_count].y > 0.53 or cylinder_pose_list[trial_count].z < 0.31:
        #     print(f"y value > 0.53 or z < 0.3, removing trial: {trial_count}")
        #     delete_trial_index.append(trial_count)

        # remove trial if the ground truth contact point z value is greater than 0.38
        if gt_contact_point_list[trial_count].z > 0.35:
            print(f"gt_contact_point z value > 0.35, removing trial: {trial_count}")
            delete_trial_index.append(trial_count)
    
    #delete the trials with y > 0.53
    for index in sorted(delete_trial_index, reverse=True):
        print(f"deleted index: {index}, value of y: {cylinder_pose_list[index].y}")
        del cylinder_pose_list[index]
        del transformed_contact_pt_list[index]
        del min_dist_list[index]
        del gt_contact_point_list[index]
    # # =====================================================
        

    #Add all contact points to vissualizer. Create a GIF of all contact points
    if CREATE_GIF_ALL_CONTACTS:
        create_prediction_visualization_gif(transformed_contact_pt_list, cylinder_pose_list, gt_contact_point_list, save_gif=False)

    # Convert a o3d pointcloud for the contact point transformed
    pcloud_of_contactpts = convert_list_of_points_to_pcloud(transformed_contact_pt_list)
 
    

    # evaluate the Chamfer Distance between the predicted point cloud and the ground truth point cloud
    CD_singleway = eval_utils.compute_chamfer_distance_singleway(pcloud_of_contactpts, collision_obj_pcloud)

    # compute MAE between the predicted contact point and the ground truth contact point
    MAE_contact_point = eval_utils.compute_MAE_contact_point(transformed_contact_pt_list, gt_contact_point_list)
    euclidean_distance, std_deviation = eval_utils.compute_euclidean_distance(transformed_contact_pt_list, gt_contact_point_list)
    MSE_contact_point = eval_utils.compute_MSE_contact_point(transformed_contact_pt_list, gt_contact_point_list)


    print(f"Chamfer Distance: {CD_singleway}, MAE: {MAE_contact_point}, Mean euclidean_distance: {euclidean_distance}, STD: {std_deviation}")

    


    #save the output prediction 
    
    print(f"**** end script **** ")
    # -------------------------------------------------------

if __name__ == '__main__':
    main()