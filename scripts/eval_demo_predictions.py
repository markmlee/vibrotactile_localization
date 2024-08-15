import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import pandas as pd
import sys

from scipy.spatial import cKDTree


"""
Load (1) predictions and (2) GT pointcloud
Transform GT pointcloud to predictions' coordinate system via using ICP, and save the transformed GT pointcloud

Load (1) predictions and (2) transformed GT pointcloud
Evaluate predictions using the transformed GT pointcloud with Chamfer Distance
Report metric and visualize 3D pointclouds
"""

def load_predictions(dir_path):
    #load predictions
    pass

def load_gt_pointcloud(dir_path):
    #load GT pointcloud
    pass

def visualize_two_pointclouds(pcd_prediction, pcd_ground_truth):
    #visualize two pointclouds
    
    # Color the point clouds
    pcd_ground_truth.paint_uniform_color([0, 0, 1])  # Blue
    pcd_prediction.paint_uniform_color([1, 0, 0])    # Red

    # Create a coordinate frame at the origin
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Visualize the point clouds together
    o3d.visualization.draw_geometries([coordinate_frame, pcd_ground_truth, pcd_prediction])




def get_transformation_ICP(pcd_prediction, pcd_ground_truth):
    """
    Get the transformation matrix to transform the GT pointcloud to the predictions' coordinate system
    input: 2 pointclouds (GT as green, predictions as red)
    output: transformation matrix

    visualize the transformed GT pointcloud and predictions
    """
    # Apply point-to-point ICP to align the prediction point cloud to the ground truth point cloud
    # Perform point-to-point ICP
    threshold = 0.1  # distance threshold for ICP convergence

    # Provided initial transformation matrix
#     initial_transform = np.array([
#     [1, 0, 0, -0.83908223],
#     [0, 1, 0, -0.65787027],
#     [0, 0, 1, 0.10417489],
#     [0, 0, 0, 1]
# ])
    
    initial_transform = np.eye(4)  # Identity matrix


    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_ground_truth, pcd_prediction, threshold,
        initial_transform,  # initial alignment
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )

    #print the transformation matrix
    print(f"Transformation matrix after ICP:\n{reg_p2p.transformation}")

    # Transform the ground truth point cloud
    pcd_ground_truth_transformed = pcd_ground_truth.transform(reg_p2p.transformation)

    # Color code the point clouds
    pcd_prediction.paint_uniform_color([1, 0, 0])  # Red for prediction
    pcd_ground_truth_transformed.paint_uniform_color([0, 1, 0])  # Green for ground truth

    # Create a coordinate frame at the origin
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])


    # Visualize the transformed ground truth point cloud and the prediction point cloud
    o3d.visualization.draw_geometries([coordinate_frame, pcd_prediction, pcd_ground_truth_transformed])

    # Return the transformation matrix
    return reg_p2p.transformation, pcd_ground_truth_transformed

def plot_2D_predictions(predictions_np):
    """
    2D scatter plot of (x,y) points which are the first 2 values of predictions numpy array.
    Color the points using gradient based on index order.
    """

    # Create a 2D scatter plot of the predictions
    plt.figure()
    plt.scatter(predictions_np[:, 0], predictions_np[:, 1], c=range(len(predictions_np)), cmap='viridis')
    plt.colorbar()
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    plt.title('2D Scatter Plot of Predictions')

    # Make x and y axes equal in scale
    plt.axis('equal')

    plt.show()

def append_trial_predictions_to_file(folder_directory, num_trial_predictions):
    """
    given a directory of many trials, crawl into each folder and append the predictions into a single file
    """
    predicted_pts = []

    for i in range(num_trial_predictions):
        pred_np = np.load(f'{folder_directory}/trial{i}/predicted_pt.npy')
        predicted_pts.append(pred_np)

    #save predictions to a single file
    predicted_pts = np.array(predicted_pts)
    
    #print the shape of the predictions
    print(f"predicted_pts shape: {predicted_pts}")

    np.save(f'{folder_directory}/contact_pts.npy', predicted_pts)

    sys.exit()


def preprocess_point_cloud(pcd, voxel_size):
    """ Downsample and compute normals and FPFH features for the point cloud. """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd_down, fpfh
 

def execute_global_registration_RANSAC(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """ Perform RANSAC-based global registration to find an initial alignment. """

    distance_threshold = 0.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down, target=target_down, source_feature=source_fpfh, target_feature= target_fpfh, mutual_filter= True,
        max_correspondence_distance= distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,  # RANSAC number of points
        checkers = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(confidence=0.9, max_iteration=4000000)
    )
    return result

def get_init_transform_via_RANSAC(pcd_prediction, pcd_ground_truth):
    """
    1) downsample pointclouds
    2) RANSAC
    return the transformation matrix
    """

    voxel_size = 0.01  # Set this according to the scale of your point cloud
    source_down, source_fpfh = preprocess_point_cloud(pcd_ground_truth, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd_prediction, voxel_size)

    # Perform global registration using RANSAC
    result_ransac = execute_global_registration_RANSAC(source_down, target_down, source_fpfh, target_fpfh, voxel_size)

    transform = result_ransac.transformation
    print(f"Initial transform via RANSAC:\n{transform}")

    # Transform and then Visualize the initial alignment
    source_down.transform(transform)
    visualize_two_pointclouds(target_down, source_down)

    return transform


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

def compute_hausdorff_distance(source, target):
    # Convert point clouds to numpy arrays
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    
    # Build KD-Trees for efficient nearest neighbor search
    source_tree = cKDTree(source_points)
    target_tree = cKDTree(target_points)
    
    # Compute distance from source to target
    distances_source_to_target, _ = target_tree.query(source_points, k=1)
    hausdorff_dist_source_to_target = np.max(distances_source_to_target)
    
    # Compute distance from target to source
    distances_target_to_source, _ = source_tree.query(target_points, k=1)
    hausdorff_dist_target_to_source = np.max(distances_target_to_source)
    
    # Hausdorff distance is the maximum of both directions
    hausdorff_distance = max(hausdorff_dist_source_to_target, hausdorff_dist_target_to_source)
    
    return hausdorff_distance

def main():
    # -------------------------------------------------------
    print(f"**** start script **** ")
    
    #transform GT pointcloud to predictions' coordinate system
    prediction_path = '/home/mark/audio_learning_project/data/test_mapping/cross_easy_fullv4'
    # prediction_path = '/home/mark/audio_learning_project/data/test_mapping/cross_easy4'

    # -------------- SINGLE USE! for appending predictions from multiple trials --------------
    # num_trial_predictions = 32
    # append_trial_predictions_to_file(prediction_path, num_trial_predictions)
    # ---------------------------------------------------------------------------

    prediction_file = os.path.join(prediction_path, 'contact_pts.npy')
    predictions_pointcloud = np.load(prediction_file)

    # plot_2D_predictions(predictions_pointcloud)
    # sys.exit()

    gt_pointcloud_path = '/home/mark/audio_learning_project/evaluation/3D_scan_GT'
    gt_pointcloud_file = os.path.join(gt_pointcloud_path, 'OBJECT 1A cross.pts')
    # gt_pointcloud = np.loadtxt(gt_pointcloud_file)
    df = pd.read_csv(gt_pointcloud_file, sep=r'\s+', header=None, comment='#', skiprows=1)

    # Convert to numpy array
    gt_pointcloud = df.values

    # Ensure the point cloud is of type float64 or float32
    gt_pointcloud = gt_pointcloud.astype(np.float64)  # or np.float32

    print(f"predictions_pointcloud shape: {predictions_pointcloud.shape}, gt_pointcloud shape: {gt_pointcloud.shape}")

    #create np into pcloud
    # Create Open3D PointCloud objects
    pcd_ground_truth = o3d.geometry.PointCloud()
    pcd_ground_truth.points = o3d.utility.Vector3dVector(gt_pointcloud[:, :3]) 

    pcd_prediction = o3d.geometry.PointCloud()
    pcd_prediction.points = o3d.utility.Vector3dVector(predictions_pointcloud[:,:3])


    # visualize_two_pointclouds(pcd_prediction, pcd_ground_truth)

    # Apply random transformation and noise
    # pcd_noisy, R, translation = apply_random_transform_and_noise(pcd_ground_truth)
    # Print the random transformation applied
    # print("Applied rotation matrix:\n", R)
    # print("Applied translation vector:\n", translation)

    # visualize_two_pointclouds(pcd_noisy, pcd_ground_truth)
    # sys.exit()



    #get initial transform matrix manually
    # init_transform = get_init_transform_via_RANSAC(pcd_prediction, pcd_ground_truth)

    init_transform = np.array([
    [-0.89943212, -0.20647158,  0.38521597,  0.87718977],
    [-0.0062678,   0.88738189,  0.46099251, -0.7204085],
    [-0.43701553,  0.41221702, -0.7994339,   0.40500757],
    [ 0,           0,           0,           1]
])


    pcd_ground_truth.transform(init_transform)


    # Get the transformation matrix
    transformation_matrix, pcd_ground_truth_transformed = get_transformation_ICP(pcd_prediction, pcd_ground_truth)

    visualize_two_pointclouds(pcd_prediction, pcd_ground_truth_transformed)

    # Compute Chamfer Distance
    chamfer_distance = compute_chamfer_distance(pcd_prediction, pcd_ground_truth_transformed)
    # Compute Hausdorff Distance
    hausdorff_distance = compute_hausdorff_distance(pcd_prediction, pcd_ground_truth_transformed)

    print(f"Chamfer Distance: {chamfer_distance}, Hausdorff Distance: {hausdorff_distance}")
    


    # -------------------------------------------------------

    #evaluate predictions using the transformed GT pointcloud with Chamfer Distance
    print(f"**** finished script **** ")

if __name__ == '__main__':
    main()