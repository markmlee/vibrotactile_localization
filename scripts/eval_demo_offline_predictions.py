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

#custom models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../learning')))
from models.CNN import CNNRegressor2D

#dataset
from datasets import AudioDataset

#custom utils
import microphone_utils as mic_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../learning')))
from transforms import to_mel_spectrogram, get_signal

"""
The purpose of this script is to load the model and predict the output of the model on the given dataset
The dataset is a directory containing the audio files and joint state files

The output should match the predicted referrence first locally in the file predicted_output_HeightRad.npy
The output should match the ground truth reference second locally in the file gt_label.npy
The output should match the predicted referrence third globally in the file predicted_pt.npy

"""

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
    model = CNNRegressor2D(cfg)
    model.load_state_dict(torch.load(os.path.join(cfg.model_directory, 'model.pth')))

    #verify if model is loaded by checking the model parameters
    # print(model)
    model.to(device)
    model.eval()

    #print length of dataset
    total_trials = len(dataset)
    print(f"total_trials: {total_trials}")

    #for item in data_loader:
    for _, (x, y, _) in enumerate(tqdm(val_loader)):
        x_input, Y_val = x.float().to(device), y.float().to(device)

        with torch.no_grad():
            Y_output = model(x_input) 

            #split prediction to height and radian
            height_pred = Y_output[:,0]

            #clip height to [-11, +11]
            height_pred = torch.clamp(height_pred, -11, 11)
            height_val = Y_val[:,0]
            height_diff = height_pred - height_val

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


    # print(f"height_pred: {height_pred}, deg_pred: {deg_pred},  height_diff: {height_diff} degree_diff: {degree_diff}")

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
                                'panda_joint1':joints[-165,0],
                                'panda_joint2':joints[-165,1],
                                'panda_joint3':joints[-165,2],
                                'panda_joint4':joints[-165,3],
                                'panda_joint5':joints[-165,4],
                                'panda_joint6':joints[-165,5],
                                'panda_joint7':joints[-165,6]})
    #pose= pose['panda_hand']
    # print("keys", pose.keys())
    for key in pose.keys():
        pose = pose[key]
    # print("pose:", pose)
    
    return pose, joints[-165,:]

def convert_contactpt_to_global(cur_contact_pt, robot, robot_joints):
    """
    Convert the contact point w.r.t to the /cylinder_origin frame to the /panda_joint1 frame

    cur_contact_pt is w.r.t to the /cylinder_origin frame, which needs to be created
    /cylinder_origin frame is translated from /panda_hand frame by [0,0,-0.155]
    Returned contact_pt_global is w.r.t to the /panda_joint1 frame
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

    return contact_pt_global


def get_stick_cylinder(path_to_stl):
    """
    For loading the single stick cylinder mesh
    """
    # Load collision object mesh
    collision_obj = o3d.io.read_triangle_mesh(path_to_stl)
    collision_obj.compute_vertex_normals()
    r = R.from_quat([0.7071, 0, 0, 0.7071])
    rz = R.from_euler('z', 2, degrees=True)
    T_object = np.eye(4)
    T_object[:3,:3] = rz.as_matrix()@r.as_matrix()
    T_object[0:3,3] = [-0.0126,0.496,0.31]
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

    # Get pre-determined transformation of scanned object to rough alignment with prediction coordinate system
    transformation_final =  manual_transformation_from_predetermined_transformation()
    # Transform the ground truth point cloud to the prediction coordinate system
    pcd_ground_truth.transform(transformation_final)

    return pcd_ground_truth

    
def create_robot_visualization_gif(trial_count, cylinders, cylinder_copy, robot, robot_joints, cur_contact_pt, num_frames=30, save_gif=False):
    """
    Create a GIF of the robot visualization with varying viewpoints
    """
    # Define window dimensions
    window_width = 800
    window_height = 600

    # Create an Open3D visualization window with explicit size
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_width, height=window_height)  # Make window invisible

    # Add geometries to the visualizer
    add_geometries_to_visualizer(vis, cylinder_copy, robot, robot_joints, cur_contact_pt)

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
            # Calculate rotation angle
            angle = np.radians(60 * math.sin(2 * math.pi * i / num_frames))  # Varies between -30 and +30 degrees
            
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

        # # Clean up temporary images
        # for i in range(num_frames):
        #     os.remove(f"temp_images/frame_{i:03d}.png")
        # os.rmdir("temp_images")

        print("GIF created: robot_visualization.gif")

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

def add_geometries_to_visualizer(vis, cylinder_copy, robot, robot_joints, cur_contact_pt):
    # Add the transformed cylinder
    vis.add_geometry(cylinder_copy)

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
        
        wire_frame = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
        wire_frame.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray color
        vis.add_geometry(wire_frame)

    # Add the collision object to the visualizer
    # collision_obj = get_stick_cylinder("/home/mark/audio_learning_project/acoustic_cylinder/franka_panda/meshes/cylinder/collision_object_dense.stl") #FOR STICK SIMPLE CASE
    gt_pointcloud_path = '/home/mark/audio_learning_project/evaluation/3D_scan_GT'
    gt_pointcloud_file = os.path.join(gt_pointcloud_path, 'OBJECT 1A cross.pts')
    collision_obj = get_pcloud_cross(gt_pointcloud_file)

    vis.add_geometry(collision_obj)


    # Add the contact point
    contact_pt_transformed = convert_contactpt_to_global(cur_contact_pt, robot, robot_joints)
    contact_pt = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
    contact_pt.paint_uniform_color([1.0, 0.0, 0.0])
    contact_pt.translate(np.array([contact_pt_transformed.x, contact_pt_transformed.y, contact_pt_transformed.z]))
    vis.add_geometry(contact_pt)

    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)


@hydra.main(version_base='1.3',config_path='../learning/configs', config_name = 'inference')
def main(cfg: DictConfig):
    # -------------------------------------------------------
    print(f"**** start script **** ")
    
    #load the robot and cylinder
    urdf_dir = '/home/mark/audio_learning_project/acoustic_cylinder/franka_panda/'
    robot, cylinder = load_robot_and_cylinder(urdf_dir)


    #create dataset from the directory
    #load the model
    #predict the output
    height_pred, x_pred, y_pred, total_trials = predict_contact_from_wavfile(cfg)

    #convert tensor to np array
    height_pred = height_pred.cpu().numpy()
    x_pred = x_pred.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    print(f"size of height_pred: {height_pred.shape}, x_pred: {x_pred.shape}, y_pred: {y_pred.shape}")

    MAE_bewteen_predictions_height, MAE_bewteen_predictions_deg, MAE_gt_height, MAE_gt_deg = compare_offline_prediction_with_saved_predictions(cfg, height_pred, x_pred, y_pred, total_trials)
    print(f"MAE_height: {MAE_bewteen_predictions_height}, MAE_deg: {MAE_bewteen_predictions_deg}, MAE_gt_height: {MAE_gt_height}, MAE_gt_deg: {MAE_gt_deg}")
    
    cur_contact_pt_list = []
    ###transform the output to robot baselink frame###
    #post process XY -> radian -> XY (to ensure projection is on the unit circle) AND (resolve wrap around issues) 
    for trial_count in range(total_trials):
        cur_contact_pt = transform_predicted_XYZ_to_EE_XYZ(x_pred[trial_count], y_pred[trial_count],height_pred[trial_count], cfg.cylinder_radius, cfg.cylinder_transform_offset)
        print(f"cur_contact_pt: {cur_contact_pt}")
        cur_contact_pt_list.append(cur_contact_pt)

        # ---------------- load the robot state from the trial directory ----------------
        trial_dir = cfg.data_dir + '/trial' + str(trial_count)
        ee_pose, robot_joints = get_arm_pose_from_robotstate(trial_dir, robot)
        
        #transform the cylinder to the EE pose
        cylinder_copy = copy.deepcopy(cylinder)
        cylinder_copy.transform(ee_pose)
        cylinder_verts = np.asarray(cylinder_copy.vertices)
        cylinder_colors  = np.asarray(cylinder_copy.vertex_colors)

        # transformed_contact_pt = visualize_robot_cylinder_stick(cylinder, cylinder_copy, robot, robot_joints, cur_contact_pt)
        create_robot_visualization_gif(trial_count, cylinder, cylinder_copy, robot, robot_joints, cur_contact_pt, save_gif=True)

        # print(f"transformed_contact_pt: {transformed_contact_pt}")
        # --------------------------------------------------------------------------------



    


    #save the output prediction 
    
    print(f"**** end script **** ")
    # -------------------------------------------------------

if __name__ == '__main__':
    main()