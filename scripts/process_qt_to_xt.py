import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from urdfpy import URDF
from scipy.spatial.transform import Rotation as R


def load_robot():
    """
    Load robot URDF once and return the robot object and joint names
    """
    urdf_path = '/home/mark/audio_learning_project/acoustic_cylinder/franka_panda/panda.urdf'
    robot = URDF.load(urdf_path)
    joint_names = [joint.name for joint in robot.joints if joint.joint_type == 'revolute']
    return robot, joint_names


def skew_to_vector(skew_matrix):
    """
    Convert a skew-symmetric matrix to its corresponding vector
    input: 3x3 skew-symmetric matrix
    output: 3x1 vector [wx, wy, wz]
    """
    return np.array([skew_matrix[2,1], skew_matrix[0,2], skew_matrix[1,0]])

def convert_qt_to_xt(qt, robot=None, joint_names=None, dt=0.01):  # 100Hz sampling
    """
    Given the robot joint trajectory, convert it to robot end effector trajectory and velocity
    input: 
        qt: (n,7) numpy array of joint positions
        robot: robot URDF object (optional)
        joint_names: list of joint names (optional)
        dt: timestep (default: 0.01s for 100Hz sampling)
    output: 
        xt: (n,7) numpy array (xyz,qx,qy,qz,qw)
        xdot: (n,6) numpy array (linear_vel[3], angular_vel[3])
    """
    # Load robot only once if not provided
    if robot is None or joint_names is None:
        robot, joint_names = load_robot()
    
    # Initialize output arrays
    n_points = qt.shape[0]
    xt = np.zeros((n_points, 7))  # position and orientation
    xdot = np.zeros((n_points, 6))  # linear and angular velocity
    
    # Store all rotation matrices
    all_R = np.zeros((n_points, 3, 3))
    
    # First pass: compute all positions and orientations
    for i in range(n_points):
        # Get joint positions for current timestep
        q = qt[i]
        
        # Create configuration dictionary
        cfg = dict(zip(joint_names, q))
        
        # Compute forward kinematics
        fk = robot.link_fk(cfg=cfg)
        T = fk[robot.link_map['panda_link8']]
        
        # Extract position
        position = T[:3, 3]
        
        # Extract rotation matrix and convert to quaternion
        rotation_matrix = T[:3, :3]
        r = R.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # Returns (x,y,z,w)
        
        # Store results
        xt[i, :3] = position
        xt[i, 3:] = quaternion
        # all_R[i] = rotation_matrix
    
    # # Second pass: compute velocities
    # # Pad positions and rotations for finite differences
    # positions_padded = np.pad(xt[:, :3], ((1, 1), (0, 0)), mode='edge')
    # R_padded = np.pad(all_R, ((1, 1), (0, 0), (0, 0)), mode='edge')
    
    # for i in range(n_points):
    #     # Linear velocity using central difference
    #     xdot[i, :3] = (positions_padded[i+2] - positions_padded[i]) / (2 * dt)
        
    #     # Angular velocity using rotation matrix derivative
    #     # R' = Ω×R -> Ω× = R'R^T -> Ω = vee(R'R^T)
    #     R_dot = (R_padded[i+2] - R_padded[i]) / (2 * dt)
    #     R_current = all_R[i]
    #     skew_omega = R_dot @ R_current.T
        
    #     # Extract angular velocity from skew matrix
    #     xdot[i, 3:] = skew_to_vector(skew_omega)
    
    return xt, xdot

def plot_compare_xt_trajectories(xt_list, xt_measured_list):
    """
    Plot comparison between calculated and measured end-effector positions for each trial
    input: 
        xt_list: list of (n,7) calculated end-effector pose arrays (using only xyz)
        xt_measured_list: list of (n,3) measured end-effector position arrays
    """
    num_trials = len(xt_list)
    
    # Calculate grid dimensions for subplots
    n_rows = int(np.ceil(np.sqrt(num_trials)))
    n_cols = int(np.ceil(num_trials / n_rows))
    
    # Create figure
    fig = plt.figure(figsize=(4*n_cols, 3*n_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Plot each trial
    for trial_idx in range(num_trials):
        xt = xt_list[trial_idx]
        xt_measured = xt_measured_list[trial_idx]
        
        # Create subplot for this trial
        ax = fig.add_subplot(n_rows, n_cols, trial_idx + 1)
        
        # Plot x, y, z positions
        time = np.arange(len(xt))
        
        # Plot calculated trajectories
        ax.plot(time, xt[:, 0], 'r-', label='x calc', alpha=0.7, linewidth=1)
        ax.plot(time, xt[:, 1], 'g-', label='y calc', alpha=0.7, linewidth=1)
        ax.plot(time, xt[:, 2], 'b-', label='z calc', alpha=0.7, linewidth=1)
        
        # Plot measured trajectories
        ax.plot(time, xt_measured[:, 0], 'r--', label='x meas', alpha=0.7, linewidth=1)
        ax.plot(time, xt_measured[:, 1], 'g--', label='y meas', alpha=0.7, linewidth=1)
        ax.plot(time, xt_measured[:, 2], 'b--', label='z meas', alpha=0.7, linewidth=1)
        
        # Add labels and title
        ax.set_title(f'Trial {trial_idx}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Position (m)')
        
        # Add legend to first subplot only
        if trial_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('End-Effector Position Comparison by Trial', fontsize=16)
    return fig

def plot_orientation(xt_list):
    """
    Plot quaternion orientations for each trial
    input: 
        xt_list: list of (n,7) calculated end-effector pose arrays (xyz,qx,qy,qz,qw)
    """
    num_trials = len(xt_list)
    
    # Calculate grid dimensions for subplots
    n_rows = int(np.ceil(np.sqrt(num_trials)))
    n_cols = int(np.ceil(num_trials / n_rows))
    
    # Create figure
    fig = plt.figure(figsize=(4*n_cols, 3*n_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Colors for each quaternion component
    colors = ['r', 'g', 'b', 'k']
    labels = ['qx', 'qy', 'qz', 'qw']
    
    # Plot each trial
    for trial_idx in range(num_trials):
        xt = xt_list[trial_idx]
        
        # Create subplot for this trial
        ax = fig.add_subplot(n_rows, n_cols, trial_idx + 1)
        
        # Plot quaternion components
        time = np.arange(len(xt))
        for i, (color, label) in enumerate(zip(colors, labels)):
            ax.plot(time, xt[:, i+3], color=color, label=label, alpha=0.7, linewidth=1)
        
        # Add labels and title
        ax.set_title(f'Trial {trial_idx}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Quaternion Value')
        
        # Add legend to first subplot only
        if trial_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set y-axis limits to [-1, 1] as quaternions should be normalized
        ax.set_ylim(-1.1, 1.1)
    
    plt.suptitle('End-Effector Orientation (Quaternions) by Trial', fontsize=16)
    return fig

def plot_compare_xdot_trajectories(xdot_list, xdot_measured_list):
    """
    Plot calculated and measured end-effector velocities side by side for each trial
    input: 
        xdot_list: list of (n,6) calculated end-effector velocities (vx,vy,vz,wx,wy,wz)
        xdot_measured_list: list of (n,3) measured end-effector linear velocities (vx,vy,vz)
    """
    num_trials = len(xdot_list)
    
    # Calculate grid dimensions for subplots
    n_rows = num_trials
    n_cols = 2  # Two columns: calculated and measured
    
    # Create figure
    fig = plt.figure(figsize=(12, 4*n_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Find global velocity limits for consistent y-axis
    v_min = float('inf')
    v_max = float('-inf')
    for xdot, xdot_measured in zip(xdot_list, xdot_measured_list):
        v_min = min(v_min, np.min(xdot[:, :3]), np.min(xdot_measured))
        v_max = max(v_max, np.max(xdot[:, :3]), np.max(xdot_measured))
    
    # Add small margin to limits
    margin = 0.1 * (v_max - v_min)
    v_min -= margin
    v_max += margin
    
    # Plot each trial
    for trial_idx in range(num_trials):
        xdot = xdot_list[trial_idx]
        xdot_measured = xdot_measured_list[trial_idx]
        time = np.arange(len(xdot))
        
        # Plot calculated velocities
        ax1 = fig.add_subplot(n_rows, n_cols, 2*trial_idx + 1)
        ax1.plot(time, xdot[:, 0], 'r-', label='vx', linewidth=1)
        ax1.plot(time, xdot[:, 1], 'g-', label='vy', linewidth=1)
        ax1.plot(time, xdot[:, 2], 'b-', label='vz', linewidth=1)
        ax1.set_title(f'Trial {trial_idx} - Calculated')
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_ylim(v_min, v_max)
        if trial_idx == 0:
            ax1.legend()
        
        # Plot measured velocities
        ax2 = fig.add_subplot(n_rows, n_cols, 2*trial_idx + 2)
        ax2.plot(time, xdot_measured[:, 0], 'r-', label='vx', linewidth=1)
        ax2.plot(time, xdot_measured[:, 1], 'g-', label='vy', linewidth=1)
        ax2.plot(time, xdot_measured[:, 2], 'b-', label='vz', linewidth=1)
        ax2.set_title(f'Trial {trial_idx} - Measured')
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.grid(True, linestyle='--', alpha=0.3)
        ax2.set_ylim(v_min, v_max)
        if trial_idx == 0:
            ax2.legend()
    
    plt.suptitle('End-Effector Velocity Comparison by Trial', fontsize=16, y=0.995)
    return fig

def plot_angular_velocity(xdot_list):
    """
    Plot computed angular velocities for each trial
    input: 
        xdot_list: list of (n,6) calculated end-effector velocities (vx,vy,vz,wx,wy,wz)
    """
    num_trials = len(xdot_list)
    
    # Calculate grid dimensions for subplots
    n_rows = int(np.ceil(np.sqrt(num_trials)))
    n_cols = int(np.ceil(num_trials / n_rows))
    
    # Create figure
    fig = plt.figure(figsize=(4*n_cols, 3*n_rows))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Colors for each angular velocity component
    colors = ['r', 'g', 'b']
    labels = ['wx', 'wy', 'wz']
    
    # Plot each trial
    for trial_idx in range(num_trials):
        xdot = xdot_list[trial_idx]
        
        # Create subplot for this trial
        ax = fig.add_subplot(n_rows, n_cols, trial_idx + 1)
        
        # Plot angular velocity components
        time = np.arange(len(xdot))
        for i, (color, label) in enumerate(zip(colors, labels)):
            ax.plot(time, xdot[:, i+3], color=color, label=label, alpha=0.7, linewidth=1)
        
        # Add labels and title
        ax.set_title(f'Trial {trial_idx}')
        ax.set_xlabel('Time step')
        ax.set_ylabel('Angular Velocity (rad/s)')
        
        # Add legend to first subplot only
        if trial_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('End-Effector Angular Velocity by Trial', fontsize=16)
    return fig

def main():
    print(f"**** start script **** ")
    
    # Load data
    # data_dir = '/home/mark/audio_learning_project/data/wood_suctionOnly_horizontal_opposite_verticalv2/'
    data_dir = '/home/mark/audio_learning_project/data/wood_T25_L42_Horizontal_v2_mini/'
    
    # Get all directory paths to trials
    dir_raw = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)])
    dir_raw = [d for d in dir_raw if d.split('/')[-1].startswith('trial')]
    len_data = len(dir_raw)
    
    count = 0
    data_dir = f"{data_dir}trial"
    dir = []
    
    print(f"data_dir: {data_dir}, len(dir): {len(dir)}, len_data: {len_data}")
    while len(dir) < len_data:
        file_name = f"{data_dir}{count}"
        if file_name in dir_raw:
            dir.append(file_name)
        count += 1
    
     # Load robot once before processing trials
    robot, joint_names = load_robot()
    print("Robot model loaded successfully")

    # Load data from each trial
    qt_list = []
    xt_list = []
    xdot_t_list = []
    xt_measured_list = []
    xdot_t_measured_list = []
    
    print(f" --------- loading data ---------")
    for trial_n in range(len_data):
        qt_for_trial = np.load(os.path.join(dir[trial_n], "q_t.npy"))
        qt_list.append(qt_for_trial)
        xt_measured_for_trial = np.load(os.path.join(dir[trial_n], "x_t.npy")) #only contains xyz
        xt_measured_list.append(xt_measured_for_trial)
        xdot_t_measured_for_trial = np.load(os.path.join(dir[trial_n], "xdot_t.npy")) #only contains xyz
        xdot_t_measured_list.append(xdot_t_measured_for_trial)

        

        # Convert qt to xt
        xt_for_trial, _ = convert_qt_to_xt(qt_for_trial, robot, joint_names)
        xt_list.append(xt_for_trial)
        # xdot_t_list.append(xdot_t_for_trial)


        # Create new array with measured positions and computed orientations
        xt_combined = np.zeros((len(xt_measured_for_trial), 7))
        xt_combined[:, :3] = xt_measured_for_trial[:,:3]  # Copy measured positions
        xt_combined[:, 3:] = xt_for_trial[:, 3:]   # Copy computed orientations

        
        # xdot_t_combined = np.zeros((len(xdot_t_measured_for_trial), 6))
        # xdot_t_combined[:, :3] = xdot_t_measured_for_trial[:,:3]
        # xdot_t_combined[:, 3:] = xdot_t_for_trial[:, 3:]


        # Save the combined data back to x_t.npy
        save_path = os.path.join(dir[trial_n], "x_t.npy")
        np.save(save_path, xt_combined)
        # print(f"Saved combined trajectory for trial {trial_n} to {save_path}")

        #print every 100 trials
        if trial_n % 100 == 0:
            print(f"Processed {trial_n} trials")
    
    # # # Plot trajectories
    # fig = plot_compare_xt_trajectories(xt_list, xt_measured_list)
    # plt.show()

    # # Plot orientation trajectories
    # fig_orient = plot_orientation(xt_list)
    # plt.show()

    # # Plot velocity trajectories
    # # Plot velocity comparison
    # fig_vel = plot_compare_xdot_trajectories(xdot_t_list, xdot_t_measured_list)
    # plt.show()
    
    # # Plot angular velocity (optional, since you don't have measured angular velocities)
    # fig_ang_vel = plot_angular_velocity(xdot_t_list)
    # plt.show()
    
    print(f"**** end script **** ")

if __name__ == '__main__':
    main()