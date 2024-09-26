import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import pandas as pd
import sys

from scipy.spatial import cKDTree

from eval_demo_predictions import manual_transformation_from_predetermined_transformation, get_transformation_ICP, visualize_two_pointclouds, load_manual_measure_of_cross

"""
This script is used to sample N points that will be used for exploration motion in the demo.
Input is the point cloud of the object.
Output is the sampled points.

Sampled points are saved in a csv file.
Sampled points are 
(1) uniformly sampled points along the Z plane of the object
(2) Filter out the points that are too close to the object surface (using diameter of the end-effector tube)
(3) Filter out the points so that only the closest layer of points are remaining (similar to thresholding by signed distance function values)

"""

def load_prediction_ground_truth_pointclouds():

    gt_pointcloud_path = '/home/mark/audio_learning_project/evaluation/3D_scan_GT'
    gt_pointcloud_file = os.path.join(gt_pointcloud_path, 'OBJECT 1A cross.pts')
    # gt_pointcloud = np.loadtxt(gt_pointcloud_file)
    df = pd.read_csv(gt_pointcloud_file, sep=r'\s+', header=None, comment='#', skiprows=1)

    # Convert to numpy array
    gt_pointcloud = df.values

    # Ensure the point cloud is of type float64 or float32
    gt_pointcloud = gt_pointcloud.astype(np.float64)  # or np.float32

    # Create Open3D PointCloud objects
    pcd_ground_truth = o3d.geometry.PointCloud()
    pcd_ground_truth.points = o3d.utility.Vector3dVector(gt_pointcloud[:, :3]) 


    # load a manual measurement of the cross in 3D space
    manual_points = load_manual_measure_of_cross()
    predictions_pointcloud = manual_points #use manual points for now

    pcd_prediction = o3d.geometry.PointCloud()
    pcd_prediction.points = o3d.utility.Vector3dVector(predictions_pointcloud[:,:3])

    

    # Get pre-determined transformation of scanned object to rough alignment with prediction coordinate system
    transformation_final =  manual_transformation_from_predetermined_transformation()
    # Transform the ground truth point cloud to the prediction coordinate system
    pcd_ground_truth.transform(transformation_final)
    # fine tune the transformation using ICP
    transformation_matrix, pcd_ground_truth_transformed = get_transformation_ICP(pcd_prediction, pcd_ground_truth)

    return pcd_prediction, pcd_ground_truth_transformed


def get_points_in_plane(pcd_ground_truth_transformed, num_points=200):
    """
    Uniformly sample points along the 2D Z plane of the object. Can assume Z = 0.315m.
    upper-left p1 = (-0.5, 0.0, 0.315)
    lower-right p2 = (+0.5, +0.5, 0.315)
    
    return sampled_points_in_plane as an Open3D PointCloud object
    """
    
    # Define the boundaries of the plane
    x_min, y_min = -0.30, 0.1
    x_max, y_max = 0.30, 0.6
    z = 0.315
    
    # Calculate the number of points in each dimension
    num_points_per_side = int(np.sqrt(num_points))
    
    # Create a grid of points
    x = np.linspace(x_min, x_max, num_points_per_side)
    y = np.linspace(y_min, y_max, num_points_per_side)
    xx, yy = np.meshgrid(x, y)
    
    # Flatten the grid and add the constant z-coordinate
    sampled_points = np.column_stack((xx.ravel(), yy.ravel(), np.full(num_points_per_side**2, z)))
    
    # Create an Open3D PointCloud object
    pcd_sampled = o3d.geometry.PointCloud()
    pcd_sampled.points = o3d.utility.Vector3dVector(sampled_points)
    
    return pcd_sampled

def get_points_inplane_nocollision(sampled_points_in_plane, pcd_ground_truth_transformed, radius=0.055, buffer_multiplier=1.3):
    """
    Given the sampled points in the plane, filter out the points that are too close to the object surface.
    Compute the closest distance from each point to the object surface, and remove the points that are too close.

    return sampled_points_filtered_close as an Open3D PointCloud object
    """
    # Convert Open3D PointCloud objects to numpy arrays
    sampled_array = np.asarray(sampled_points_in_plane.points)
    ground_truth_array = np.asarray(pcd_ground_truth_transformed.points)

    # Create KD-Tree from the ground truth point cloud
    tree = cKDTree(ground_truth_array)

    # Compute distances from each sampled point to the nearest point in the ground truth
    distances, _ = tree.query(sampled_array)

    # Calculate the threshold distance
    threshold = radius * buffer_multiplier

    # Create a boolean mask for points that are far enough away
    mask = distances > threshold

    # Filter the points
    filtered_points = sampled_array[mask]

    # Create a new Open3D PointCloud object with the filtered points
    sampled_points_filtered_close = o3d.geometry.PointCloud()
    sampled_points_filtered_close.points = o3d.utility.Vector3dVector(filtered_points)

    return sampled_points_filtered_close

def get_points_inplane_nocollision_closest(sampled_points_filtered_close, pcd_ground_truth_transformed):
    """
    Given the sampled points in the plane, filter out the points so that only the closest layer of points are remaining.
    The closest layer of points are the points that are closest to the object surface while maintaining the approximate shape or boundary of the object.
    As such, use the concepts like signed distance function values to maintain the boundary of the object while filter out far points.
    """
    # Convert Open3D PointCloud objects to numpy arrays
    sampled_array = np.asarray(sampled_points_filtered_close.points)
    ground_truth_array = np.asarray(pcd_ground_truth_transformed.points)

    # Create KD-Tree from the ground truth point cloud
    tree = cKDTree(ground_truth_array)

    # Compute distances from each sampled point to the nearest point in the ground truth
    distances, _ = tree.query(sampled_array)

    # Compute the mean and standard deviation of the distances
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Define a threshold based on the mean and standard deviation
    # Points within this threshold will be considered as the "closest layer"
    threshold = mean_distance - 0.5 * std_distance
    threshold = 0.1

    # Create a boolean mask for points that are within the threshold
    mask = distances <= threshold

    # Filter the points
    filtered_points = sampled_array[mask]

    # Create a new Open3D PointCloud object with the filtered points
    sampled_points_filtered_closest_layer = o3d.geometry.PointCloud()
    sampled_points_filtered_closest_layer.points = o3d.utility.Vector3dVector(filtered_points)

    return sampled_points_filtered_closest_layer

def save_sampled_points(sampled_points_filtered_closest_layer):
    """
    Save the sampled points to a npy file.
    """
    # Convert the Open3D PointCloud object to a numpy array
    sampled_array = np.asarray(sampled_points_filtered_closest_layer.points)

    #save directory
    save_dir = '/home/mark/audio_learning_project/data/test_mapping'
    
    # Save the numpy array to a npy file 
    np.save(os.path.join(save_dir, 'sampled_points.npy'), sampled_array)
    
    #print current absolute directory path of saved file
    print(f"Saved sampled points to: {os.path.abspath('sampled_points.npy')}")

def get_points_inplane_nocollision_closest_validrobot(sampled_points_filtered_closest_layer, pcd_ground_truth_transformed):
    """
    filter out points that are not valid for the robot to reach.
    Heuristic: in XY plane, x[-0.25, +0.25] y[+0.2, +0.55]
    """

    # Define the boundaries of the valid robot workspace
    x_min, x_max = -0.22, 0.22
    y_min, y_max = 0.2, 0.55

    # Convert Open3D PointCloud objects to numpy arrays
    sampled_array = np.asarray(sampled_points_filtered_closest_layer.points)

    # Create a boolean mask for points that are within the valid robot workspace
    mask = (sampled_array[:, 0] >= x_min) & (sampled_array[:, 0] <= x_max) & (sampled_array[:, 1] >= y_min) & (sampled_array[:, 1] <= y_max)

    # Filter the points
    filtered_points = sampled_array[mask]

    # Create a new Open3D PointCloud object with the filtered points
    sampled_points_filtered_closest_layer_validrobot = o3d.geometry.PointCloud()
    sampled_points_filtered_closest_layer_validrobot.points = o3d.utility.Vector3dVector(filtered_points)

    return sampled_points_filtered_closest_layer_validrobot

def main():
    # -------------------------------------------------------
    print(f"**** start script **** ")

    pcd_prediction, pcd_ground_truth_transformed = load_prediction_ground_truth_pointclouds()

    VISUALIZE = True

    if VISUALIZE: visualize_two_pointclouds(pcd_prediction, pcd_ground_truth_transformed)

    # uniformly sample points along the Z plane of the object
    sampled_points_in_plane = get_points_in_plane(pcd_ground_truth_transformed, 500)
    print(f"Init sampled pts: {len(sampled_points_in_plane.points)}")

    #visualize the sampled points
    if VISUALIZE: visualize_two_pointclouds(sampled_points_in_plane, pcd_ground_truth_transformed)

    # Filter out the points that are too close to the object surface (using diameter of the end-effector tube)
    sampled_points_filtered_close = get_points_inplane_nocollision(sampled_points_in_plane, pcd_ground_truth_transformed)
    print(f"Removed collision. Remaining: {len(sampled_points_filtered_close.points)}")

    #visualize the sampled points
    if VISUALIZE: visualize_two_pointclouds(sampled_points_filtered_close, pcd_ground_truth_transformed)

    # Filter out the points so that only the closest layer of points are remaining (similar to thresholding by signed distance function values)
    sampled_points_filtered_closest_layer = get_points_inplane_nocollision_closest(sampled_points_filtered_close, pcd_ground_truth_transformed)
    print(f"Got closest. Remaining: {len(sampled_points_filtered_closest_layer.points)}")

    # visualize the remaining points with GT point cloud
    if VISUALIZE: visualize_two_pointclouds(sampled_points_filtered_closest_layer, pcd_ground_truth_transformed)

    # Filter out the points so that only the closest layer of points are remaining (similar to thresholding by signed distance function values)
    sampled_points_filtered_closest_layer_validrobot = get_points_inplane_nocollision_closest_validrobot(sampled_points_filtered_closest_layer, pcd_ground_truth_transformed)
    print(f"Removed robot range. Remaining: {len(sampled_points_filtered_closest_layer_validrobot.points)}")
    
    # visualize the remaining points with GT point cloud
    if VISUALIZE: visualize_two_pointclouds(sampled_points_filtered_closest_layer_validrobot, pcd_ground_truth_transformed)

    #downsample points sampled_points_filtered_closest_layer_validrobot to be more efficient
    #sampled_points_filtered_closest_layer_validrobot = sampled_points_filtered_closest_layer_validrobot.voxel_down_sample(voxel_size=0.05)
    #print(f"Downsampled to {len(sampled_points_filtered_closest_layer_validrobot.points)}")

    # visualize the remaining points with GT point cloud
    #if VISUALIZE: visualize_two_pointclouds(sampled_points_filtered_closest_layer_validrobot, pcd_ground_truth_transformed)
    

    # save the sampled points
    save_sampled_points(sampled_points_filtered_closest_layer_validrobot)



    # -------------------------------------------------------
    print(f"**** finished script **** ")

if __name__ == '__main__':
    main()