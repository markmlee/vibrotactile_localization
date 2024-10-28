import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import pandas as pd
import sys

from scipy.spatial import cKDTree
from geometry_msgs.msg import Point

def convert_h_rad_to_xyz(height, radian, radius):
    """
    Convert height and radian to x, y, z coordinates.
    Return Point() object.
    """
    x = radius * np.cos(radian)
    y = radius * np.sin(radian)
    z = height

    return Point(x, y, z)


def compute_MAE_contact_point(source, target):
    """
    Given input Point() objects, compute the mean absolute error (MAE) between the contact points.
    Assume indices are aligned.
    """
    # Convert lists of points to numpy arrays
    source_points = np.array([[pt.x, pt.y, pt.z] for pt in source])
    target_points = np.array([[pt.x, pt.y, pt.z] for pt in target])

    # Compute the mean absolute error (MAE) between the contact points
    mae = np.mean(np.abs(source_points - target_points))

    diff = source_points - target_points
    print(f"diff: {diff}")

    abs_diff = np.abs(diff)
    print(f"abs_diff: {abs_diff}")

    mean_abs_diff = np.mean(abs_diff, axis=0)
    print(f"mean_abs_diff: {mean_abs_diff}")

    return mae

def compute_euclidean_distance(source, target):
    """
    Given input lists of Point() objects, compute the Euclidean distance between the contact points.
    Assume indices are aligned.
    """
    # Convert lists of points to numpy arrays
    source_points = np.array([[pt.x, pt.y, pt.z] for pt in source])
    target_points = np.array([[pt.x, pt.y, pt.z] for pt in target])

    # Compute the Euclidean distance between the contact points
    distances = np.linalg.norm(source_points - target_points, axis=1)

    # Compute mean and standard deviation
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    return mean_distance, std_distance

def compute_MSE_contact_point(source, target):
    """
    Given input lists of Point() objects, compute the mean squared error (MSE) between the contact points.
    Assume indices are aligned.
    """
    # Convert lists of points to numpy arrays
    source_points = np.array([[pt.x, pt.y, pt.z] for pt in source])
    target_points = np.array([[pt.x, pt.y, pt.z] for pt in target])

    # Compute the mean squared error (MSE) between the contact points
    mse = np.mean(np.square(source_points - target_points))

    return mse

def compute_chamfer_distance_singleway(source, target):
    # Convert point clouds to numpy arrays
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    
    # Build KD-Trees for efficient nearest neighbor search
    source_tree = cKDTree(source_points)
    target_tree = cKDTree(target_points)
    
    # Compute distance from source to target
    distances_source_to_target, _ = target_tree.query(source_points, k=1)
    chamfer_dist_source_to_target = np.mean(np.square(distances_source_to_target))


    # Compute distance from target to source
    distances_target_to_source, _ = source_tree.query(target_points, k=1)
    chamfer_dist_target_to_source = np.mean(np.square(distances_target_to_source))
    
 
    # Chamfer distance is the sum of both directions
    chamfer_distance = chamfer_dist_source_to_target #####+ chamfer_dist_target_to_source
    chamfer_distance = np.sqrt(chamfer_distance) ###ADDED SQRT to look at distance only (unit in meters)
    
    return chamfer_distance 



def apply_random_transform_and_noise(pcd, noise_std=0.01, translation_range=1.0, rotation_range=np.pi/4):
    """
    Apply a random transformation (rotation and translation) and Gaussian noise to the input point cloud.
    
    :param pcd: The input point cloud (Open3D PointCloud object).
    :param noise_std: Standard deviation of the Gaussian noise to be added.
    :param translation_range: Maximum range of the random translation (in meters).
    :param rotation_range: Maximum rotation angle (in radians).
    :return: Transformed and noisy point cloud (Open3D PointCloud object).
    """
    
    # Step 1: Generate a random rotation matrix
    # Random angles for rotation
    theta_x = np.random.uniform(-rotation_range, rotation_range)
    theta_y = np.random.uniform(-rotation_range, rotation_range)
    theta_z = np.random.uniform(-rotation_range, rotation_range)
    
    # Rotation matrices around x, y, z axes
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(theta_x), -np.sin(theta_x)],
                   [0, np.sin(theta_x), np.cos(theta_x)]])
    
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                   [0, 1, 0],
                   [-np.sin(theta_y), 0, np.cos(theta_y)]])
    
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                   [np.sin(theta_z), np.cos(theta_z), 0],
                   [0, 0, 1]])
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    
    # Step 2: Generate a random translation vector
    translation = np.random.uniform(-translation_range, translation_range, size=(3,))
    
    # Step 3: Apply the transformation to the point cloud
    points = np.asarray(pcd.points)
    transformed_points = np.dot(points, R.T) + translation
    
    # Step 4: Add Gaussian noise
    noise = np.random.normal(0, noise_std, transformed_points.shape)
    noisy_points = transformed_points + noise
    
    # Step 5: Create a new point cloud with the transformed and noisy points
    pcd_transformed_noisy = o3d.geometry.PointCloud()
    pcd_transformed_noisy.points = o3d.utility.Vector3dVector(noisy_points)
    
    # Copy colors (if any) from the original point cloud
    if pcd.has_colors():
        pcd_transformed_noisy.colors = pcd.colors
    
    return pcd_transformed_noisy, R, translation

def compute_chamfer_distance(source, target):
    # Convert point clouds to numpy arrays
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    
    # Build KD-Trees for efficient nearest neighbor search
    source_tree = cKDTree(source_points)
    target_tree = cKDTree(target_points)
    
    # Compute distance from source to target
    distances_source_to_target, _ = target_tree.query(source_points, k=1)
    chamfer_dist_source_to_target = np.mean(np.square(distances_source_to_target))


    # Compute distance from target to source
    distances_target_to_source, _ = source_tree.query(target_points, k=1)
    chamfer_dist_target_to_source = np.mean(np.square(distances_target_to_source))
    
 
    # Chamfer distance is the sum of both directions
    chamfer_distance = chamfer_dist_source_to_target + chamfer_dist_target_to_source
    
    return chamfer_distance 