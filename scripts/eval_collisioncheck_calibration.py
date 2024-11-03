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

#custom models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../learning')))
from models.CNN import CNNRegressor2D

#dataset
from datasets import AudioDataset

#custom utils
import microphone_utils as mic_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../learning')))
from transforms import to_mel_spectrogram, get_signal
import eval_utils as eval_utils

    
#collision
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
from scipy.spatial import KDTree

 # ==============================================================================================================
class Checker_Collision:
    def __init__(self, cylinder, collision_obj_pcloud, robot, robot_joints):
        self.cylinder = cylinder
        self.collision_obj_pcloud = collision_obj_pcloud
        self.robot = robot
        self.robot_joints = robot_joints
        self.vis = None
        self.robot_geometries = None

    def setup_visualization(self, window_width=800, window_height=600):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=window_width, height=window_height)
        self._add_geometries_to_visualizer()
        # self.robot_geometries = self._add_robot_to_visualizer()
        self._set_camera_view()

    def _add_geometries_to_visualizer(self):
        # Add the collision object point cloud
        self.vis.add_geometry(self.collision_obj_pcloud)
        
        # Add coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        self.vis.add_geometry(coordinate_frame)

    def _add_robot_to_visualizer(self):
        robot_geometries = []
        visual_geometries = self.robot.visual_trimesh_fk(cfg={
            f'panda_joint{i+1}': self.robot_joints[i] for i in range(7)
        })

        for trimesh_obj, transform in visual_geometries.items():
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(trimesh_obj.vertices))
            o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(trimesh_obj.faces))
            o3d_mesh.transform(transform)
            o3d_mesh.compute_vertex_normals()
            o3d_mesh.paint_uniform_color([0.8, 0.8, 0.8])
            self.vis.add_geometry(o3d_mesh)
            robot_geometries.append(o3d_mesh)

        return robot_geometries

    def _set_camera_view(self):
        ctr = self.vis.get_view_control()
        ctr.set_front([-0.75, 2, 0.5])
        ctr.set_up([0.4, 0.15, 1.5])
        ctr.set_lookat([0, 0, 0.5])
        ctr.set_zoom(0.85)

    def update_robot_position(self, robot_joints):

        new_visual_geometries = self.robot.visual_trimesh_fk(cfg={
            f'panda_joint{i+1}': robot_joints[i] for i in range(7)
        })


        for geom, (trimesh_obj, transform) in zip(self.robot_geometries, new_visual_geometries.items()):
            geom.vertices = o3d.utility.Vector3dVector(np.asarray(trimesh_obj.vertices))
            geom.triangles = o3d.utility.Vector3iVector(np.asarray(trimesh_obj.faces))
            geom.transform(transform)
            geom.compute_vertex_normals()
            self.vis.update_geometry(geom)

    def find_closest_point(self, cylinder_pts, pcloud_pts, threshold=0.01):
        target_kdtree = KDTree(pcloud_pts)
        close_distances, close_indices = target_kdtree.query(cylinder_pts, distance_upper_bound=threshold)

        valid_indices = close_distances < threshold
        if not np.any(valid_indices):
            closest_index = np.argmin(close_distances)
            return closest_index, close_distances[closest_index], cylinder_pts[closest_index]

        close_distances = close_distances[valid_indices]
        close_cylinder_points = cylinder_pts[valid_indices]

        weights = 1 / (close_distances + 1e-6)
        weights /= np.sum(weights)

        avg_point = np.average(close_cylinder_points, axis=0, weights=weights)
        cylinder_kdtree = KDTree(cylinder_pts)
        closest_distance, closest_index = cylinder_kdtree.query(avg_point.reshape(1, -1))

        return closest_index[0], closest_distance[0], cylinder_pts[closest_index[0]]

    def get_arm_pose_from_robotstate(self, joint_states):
        pose = self.robot.link_fk(links=['panda_hand'], cfg={
            f'panda_joint{i+1}': joint_states[i] for i in range(7)
        })
        
        # Assuming pose contains a single Link object
        if len(pose) == 1:
            return next(iter(pose.values())), joint_states
        else:
            # If there's more than one link, iterate through them
            for link in pose.values():
                return link, joint_states  # Return the first link found
        
        # If no pose is found
        raise ValueError("No valid pose found in the forward kinematics result")

    def visualize_contact_point(self, contact_point):
        # Create a red sphere for the contact point
        contact_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)  # Adjust the radius as needed
        contact_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
        contact_sphere.translate(contact_point)

        return contact_sphere
        

    def update_visualization(self):
        self.vis.poll_events()
        self.vis.update_renderer()

    def capture_image(self):
        return np.asarray(self.vis.capture_screen_float_buffer(False))

    def run_collision_check(self, joint_states_list, visualize=False):
        min_dist_list = []
        image_list = []
        contact_point_list = []
        pcloud_pts = np.asarray(self.collision_obj_pcloud.points)

        # Variables to keep track of the geometries added in the previous iteration
        previous_contact_sphere = None
        previous_wire_frame = None

        for joint_states in joint_states_list:
            
            # Update robot position            
            # self.update_robot_position(joint_states)

            # Get new end-effector pose
            ee_pose, _ = self.get_arm_pose_from_robotstate(joint_states)

            # Transform cylinder to new end-effector pose
            cylinder_copy = copy.deepcopy(self.cylinder)
            cylinder_copy.transform(ee_pose)
            cylinder_pts = np.asarray(cylinder_copy.vertices)
            cylinder_colors = np.asarray(cylinder_copy.vertex_colors)

            # Find closest point
            idx, min_dist, contact_point = self.find_closest_point(cylinder_pts, pcloud_pts)
            min_dist_list.append(min_dist)

            #create into a point
            contact_point_pt = Point()
            contact_point_pt.x = contact_point[0]
            contact_point_pt.y = contact_point[1]
            contact_point_pt.z = contact_point[2]
            contact_point_list.append(contact_point_pt)

            # print(f"Closest index: {idx}, min_dist: {min_dist}")

            if visualize:
                # Remove previous geometries
                if previous_contact_sphere is not None:
                    self.vis.remove_geometry(previous_contact_sphere)
                if previous_wire_frame is not None:
                    self.vis.remove_geometry(previous_wire_frame)

                # Visualize contact point
                contact_sphere = self.visualize_contact_point(contact_point)
                self.vis.add_geometry(contact_sphere)
                previous_contact_sphere = contact_sphere

                # Visualize cylinder
                wire_frame = o3d.geometry.LineSet.create_from_triangle_mesh(cylinder_copy)
                self.vis.add_geometry(wire_frame)
                previous_wire_frame = wire_frame

                # Update visualization
                self.update_visualization()

                # Capture image
                image = self.capture_image()
                image_list.append((image * 255).astype(np.uint8))

                time.sleep(0.5)

        return min_dist_list, image_list, contact_point_list

    def close_visualization(self):
        self.vis.destroy_window()



    # ==============================================================================================================


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

    # Create an additional translation matrix for 3cm in Z direction
    manual_translation = np.eye(4)
    manual_translation[1, 3] = 0.03  # 3cm down
    # manual_translation[0, 3] = -0.02  # 2cm right 

    # Combine the transformations
    combined_transformation = manual_translation @ transformation_final

    # Transform the ground truth point cloud to the prediction coordinate system
    pcd_ground_truth.transform(combined_transformation)

    return pcd_ground_truth


def add_collision_object_to_visualizer(vis):
    """
    Add the collision object to the visualizer
    Input: path to the collision object STL file
    """

    # Add the collision object to the visualizer
    # collision_obj = get_stick_cylinder("/home/mark/audio_learning_project/acoustic_cylinder/franka_panda/meshes/cylinder/collision_object_dense.stl") #FOR STICK SIMPLE CASE
    gt_pointcloud_path = '/home/mark/audio_learning_project/evaluation/3D_scan_GT'
    gt_pointcloud_file = os.path.join(gt_pointcloud_path, 'OBJECT 1A cross.pts')
    collision_obj = get_pcloud_cross(gt_pointcloud_file)

    # Downsample the point cloud
    voxel_size = 0.001  # Adjust the voxel size as needed
    collision_obj_downsampled = collision_obj.voxel_down_sample(voxel_size)

    # Set the color of the point cloud to neon green
    colors = np.tile([0.0, 1.0, 0.0], (len(collision_obj_downsampled.points), 1))  # Neon green color
    collision_obj_downsampled.colors = o3d.utility.Vector3dVector(colors)

    vis.add_geometry(collision_obj_downsampled)

    return collision_obj #collision_obj_downsampled


def add_geometries_to_visualizer(vis):
    """
    Update the visualizer with the transformed cylinder, collision object, and contact point
    """

    # Add the collision object (path is hardcoded for now)
    collision_obj_pcloud = add_collision_object_to_visualizer(vis)

    

    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    return collision_obj_pcloud
    


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




def get_arm_pose_from_robotstate(robot, joint_states):
    """
    Load the robot state from the trial directory, do FK and return the pose of the end-effector
    Returns a 4x4 Trans matrix
    """

    joints = joint_states
    pose= robot.link_fk(links=['panda_hand'], cfg={
                                'panda_joint1':joints[0],
                                'panda_joint2':joints[1],
                                'panda_joint3':joints[2],
                                'panda_joint4':joints[3],
                                'panda_joint5':joints[4],
                                'panda_joint6':joints[5],
                                'panda_joint7':joints[6]})
    #pose= pose['panda_hand']
    # print("keys", pose.keys())
    for key in pose.keys():
        pose = pose[key]
    # print("pose:", pose)

    
    return pose, joints


def get_joint_states():

    joint_states = [
    [ 1.26876024, -1.14595771, -0.03479162, -2.72549357,  0.03621171,  1.59198118, -1.71029246],
    [ 1.3950849,  -0.92396451, -0.01047735, -2.69677565, -0.02460548,  1.77789074, -1.8935403 ],
    [ 1.36985452, -0.79462138,  0.01839074, -2.67184609,  0.01053125,  1.90848686, -1.89141126],
    [ 1.34321256, -0.60677455,  0.01720362, -2.57601769,  0.01049919,  1.98256302, -1.89173362],
    [ 1.24502217e+00, -6.07715207e-01,  2.52506429e-03, -2.54845123e+00, -2.84848328e-03,  1.98678215e+00, -1.89308469e+00],
    [ 1.12705622, -0.85621646, -0.02296481, -2.79086311,  0.00681183,  2.29103464, -2.1225794 ],
    [ 0.95913577,  0.00758971,  0.22506569, -1.81627146, -0.03544656,  1.69354878, -2.13321726],
    [ 0.9900899,   0.00773281,  0.27728221, -1.81717802, -0.01989263,  1.6462941,  -2.13215108],
    [ 1.01359978e+00,  1.07274157e-01,  3.49341789e-01, -1.82769409e+00, -1.96625364e-03,  1.59153929e+00, -2.13209948e+00],
    [ 1.15365585, -0.14577625,  0.28236266, -2.07106478,  0.04726823,  1.92808164, -2.12449605],
    [ 1.14294075, -0.026576,    0.30287374, -2.02980799,  0.04645126,  2.02922029, -2.11006177],
    [ 1.13872687,  0.09130928,  0.31802471, -1.88254584,  0.03408398,  2.00733301, -2.10918471],
    [ 1.07602368,  0.08537513,  0.59183514, -1.77812332,  0.0593767,   1.90503417, -2.10247535],
    [ 1.86381789, -0.13046835, -0.12220333, -1.93333992, -0.04441785,  1.78522508, -2.06835709],
    [ 1.88095908,  0.18700402,  0.00468935, -1.66215578, -0.10354735,  1.47192335, -2.07968668],
    [ 1.90857513,  0.18894129,  0.07672438, -1.65622377, -0.07837959,  1.52854417, -2.00121445],
    [ 1.82318355, -0.62300629,  0.25904728, -2.44887709,  0.05901053,  1.98538649, -2.34097094],
    [ 1.68671102, -0.62829478,  0.22026915, -2.45017876,  0.06040149,  1.88898025, -2.33944027],
    [ 1.64265778, -0.68585559,  0.1711383,  -2.54592787,  0.04702472,  1.83763568, -2.37645483],
    [ 1.61639128, -1.0216631,   0.17886272, -2.71479599,  0.16950175,  1.66266311, -2.37561468]
    ]
    
    return joint_states

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

def find_closest_point_index(target_points, source_points, threshold=0.05):
    """
    Find the most likely point of contact between two sets of points.

    Parameters:
    target_points (numpy.ndarray): Array of target mesh points [nx3].
    source_points (numpy.ndarray): Array of source mesh points [mx3].
    threshold (float): Distance threshold for considering points as intersecting.

    Returns:
    int: Index of the most likely contact point in source_points.
    """
    # Create KDTree for target points
    target_kdtree = KDTree(target_points)

    # Find all points within the threshold distance
    close_distances, close_indices = target_kdtree.query(source_points, distance_upper_bound=threshold)

    # Filter out points that are beyond the threshold
    valid_indices = close_distances < threshold
    close_distances = close_distances[valid_indices]
    close_indices = close_indices[valid_indices]
    close_source_points = source_points[valid_indices]

    if len(close_distances) == 0:
        distances, indices = target_kdtree.query(source_points)
        closest_index = np.argmin(distances)
        return closest_index, distances[closest_index]

    # Calculate weights based on inverse distance
    weights = 1 / (close_distances + 1e-6)  # Add small epsilon to avoid division by zero
    weights /= np.sum(weights)  # Normalize weights

    # Calculate the weighted average point
    avg_point = np.average(close_source_points, axis=0, weights=weights)

    # Find the source point closest to the weighted average
    source_kdtree = KDTree(source_points)
    closest_distance, closest_index = source_kdtree.query(avg_point.reshape(1, -1))

    return closest_index[0], closest_distance[0]



def main():

    # ========================================================
    print(f"**** start script **** ")

    # Load the robot and cylinder
    urdf_dir = '/home/mark/audio_learning_project/acoustic_cylinder/franka_panda/'
    robot, cylinder = load_robot_and_cylinder(urdf_dir)

    # Get collision object point cloud
    gt_pointcloud_path = '/home/mark/audio_learning_project/evaluation/3D_scan_GT'
    gt_pointcloud_file = os.path.join(gt_pointcloud_path, 'OBJECT 1A cross.pts')
    collision_obj_pcloud = get_pcloud_cross(gt_pointcloud_file)

    # Get joint states
    joint_states = get_joint_states()
    num_trials = len(joint_states)
    print(f"num_trials: {num_trials}")

    # Initialize Checker_Collision
    initial_joints = joint_states[0]
    checker = Checker_Collision(cylinder, collision_obj_pcloud, robot, initial_joints)

    # Setup visualization
    checker.setup_visualization(window_width=800, window_height=600)

    # Run collision check
    print(f"joint_states: {joint_states}")
    min_dist_list, image_list, contact_point_list = checker.run_collision_check(joint_states, visualize=True)

    #remove inf from min_dist_list
    min_dist_list = [x for x in min_dist_list if x != np.inf]

    # Calculate average minimum distance
    avg_min_dist = np.mean(min_dist_list)
    print(f"Average minimum distance: {avg_min_dist}")

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

    # Close the visualization window
    checker.close_visualization()

    print(f"**** end script **** ")

    # ========================================================

    # # Load the robot and cylinder
    # urdf_dir = '/home/mark/audio_learning_project/acoustic_cylinder/franka_panda/'
    # robot, cylinder = load_robot_and_cylinder(urdf_dir)

    # # Define window dimensions
    # window_width = 800
    # window_height = 600

    # # Create an Open3D visualization window with explicit size
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(width=window_width, height=window_height)

    # # Add geometries to the visualizer
    # collision_obj_pcloud = add_geometries_to_visualizer(vis)

    # joint_states = get_joint_states()
    # num_trials = len(joint_states)
    # print(f"num_trials: {num_trials}")

    # # Add initial robot mesh
    # initial_pose, initial_joints = get_arm_pose_from_robotstate(robot, joint_states[0])
    # robot_geometries = add_robot_to_visualizer(vis, robot, initial_joints)

    # # Base camera parameters
    # base_front = np.array([-0.75, 2, 0.5])
    # up = [0.4, 0.15, 1.5]
    # lookat = [0, 0, 0.5]
    # zoom = 0.85

    # # Set camera view
    # ctr = vis.get_view_control()
    # ctr.set_front(base_front)
    # ctr.set_up(up)
    # ctr.set_lookat(lookat)
    # ctr.set_zoom(zoom)

    # #  -------------------------------------------------------for visualization GUI  -------------------------------------------------------
    # # trial_count = 0
    # # ee_pose, robot_joints = get_arm_pose_from_robotstate(robot, joint_states[trial_count])
    # # update_robot_position(vis, robot, robot_joints, robot_geometries)
    
    # # vis.poll_events()
    # # vis.update_renderer()
    # # vis.run()
    # # vis.destroy_window()
    # # sys.exit()
    # #  -------------------------------------------------------for visualization GUI  ------------------------------------------------

    # #average min_distance
    # avg_min_dist = 0.0
    # min_dist_list = []
    # image_list = []

    # # Run the visualization loop
    # for trial_count in range(num_trials):
    #     #get the pose of the robot and cylinder
    #     ee_pose, robot_joints = get_arm_pose_from_robotstate(robot, joint_states[trial_count])

    #     #put cylinder in the end effector pose
    #     cylinder_copy = copy.deepcopy(cylinder)
    #     cylinder_copy.transform(ee_pose)
    #     cylinder_verts = np.asarray(cylinder_copy.vertices)
    #     cylinder_colors  = np.asarray(cylinder_copy.vertex_colors)

    #     #check collision

    #     #np array of the point cloud and mesh
    #     pcloud_pts = np.asarray(collision_obj_pcloud.points)
    #     cylinder_pts = np.asarray(cylinder_copy.vertices)

    #     #print dimension of the point cloud and mesh
    #     # print(f"val pcloud: {pcloud_pts}, val vertices: {cylinder_pts}")
    #     # print(f"dim of pcloud: {pcloud_pts.shape}, dim of vertices: {cylinder_pts.shape}")

    #     idx, min_dist = find_closest_point_index(pcloud_pts, cylinder_pts)
    #     print(f"trial {trial_count}, closest index: {idx}, min_dist: {min_dist}")
    #     min_dist_list.append(min_dist)
    #     if idx == None:        
    #         print("No collision detected")

    #     # visualization of the contact pt
    #     pt = cylinder_verts[idx,:]
    #     # reshape [3] to [1,3]
    #     pt = pt.reshape(1,3)

    #     #Add contact points to visualization
    #     contact_pcd = o3d.geometry.PointCloud()
    #     contact_pcd.points = o3d.utility.Vector3dVector(np.array(pt))
    #     contact_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
    #     vis.add_geometry(contact_pcd)


    #     # visualization of robot
    #     update_robot_position(vis, robot, robot_joints, robot_geometries)
    #     vis.poll_events()
    #     vis.update_renderer()

    #     # Save the current frame as an image
    #     image = vis.capture_screen_float_buffer(False)
    #     image_np = np.asarray(image)
    #     image_list.append((image_np * 255).astype(np.uint8))
        
    #     # Optional: add a small delay to control the animation speed
    #     time.sleep(0.5)

    # # Save the last image
    # last_image = image_list[-1]
    # imageio.imwrite('last_frame.png', last_image)

    # # Create gif from images
    # # Convert fps to duration in milliseconds (e.g., 2 fps = 500 ms per frame)
    # duration = int(1000 / 2)  # 2 fps
    # imageio.mimsave('animation.gif', image_list, duration=duration)


    # # Close the window after all trials
    # vis.destroy_window()

    # #average min_distance
    # avg_min_dist = np.mean(min_dist_list)
    # print(f"avg_min_dist: {avg_min_dist}")

    # print(f"**** end script **** ")
    # -------------------------------------------------------

if __name__ == '__main__':
    main()