import matplotlib.pyplot as plt
import numpy as np
import sys
from autolab_core import RigidTransform

def plot_entire_state():
    load_file_x = '/home/iam-lab/audio_localization/vibrotactile_localization/data/franka_2D_localization/recorded_ee_pose.npy'

    ee_t_list = np.load(load_file_x, allow_pickle=True)
    print(f"size of recorded_ee_pose: {len(ee_t_list)}") #--> 100 * recording duration of entire motion

    #take only first 1/10th of the data
    shorthened_length_ratio = 2.0
    shorthened_length = int(len(ee_t_list)/shorthened_length_ratio)
    ee_t_list = ee_t_list[:shorthened_length]

    t = np.arange(0, len(ee_t_list), 1)

    translation_t = np.zeros((len(ee_t_list), 3))
    rotation_t = np.zeros((len(ee_t_list), 3, 3))
    euler_t = np.zeros((len(ee_t_list), 3))

    print(f"shape rotation_t: {rotation_t.shape}")
    
    for i in range(len(ee_t_list)):
        #ee is 4x4 matrix
        matrix4x4 = ee_t_list[i]

        #reshape list of 16 into 4x4 matrix
        matrix4x4_np = np.reshape(matrix4x4, (4,4))
        # print(f"matrix4x4_np: {matrix4x4_np}")


        #get translation from [0,1,2 col of 4th row]
        translation_t[i] = matrix4x4_np[3,0:3]
        # print(f"translation_t[i]: {translation_t[i]}")
        

        #get rotation from [0,1,2 row of 0,1,2 column]
        rotation_t[i] = matrix4x4_np[0:3,0:3]

        rt = RigidTransform(translation= translation_t[i],rotation=rotation_t[i], from_frame='world', to_frame='ee')

        #convert rotation matrix to euler angles
        euler_angles = rt.euler
        # print(f"euler angles: {euler_angles}") #--> [roll, pitch, yaw]
        euler_t[i] = euler_angles

    #plot 3x1 subplot of x_t, y_t, z_t
    fig, axs = plt.subplots(3, 1)
    fig.suptitle('Recorded EE trajectory')

    #apply same scale for all 3 subplots
    axs[0].set_ylim(0., 0.2)
    axs[1].set_ylim(0.2, 0.4)
    axs[2].set_ylim(0.3, 0.5)

    axs[0].plot(t, translation_t[:,0], label='x')
    axs[1].plot(t, translation_t[:,1], label='y')
    axs[2].plot(t, translation_t[:,2], label='z')

    #label axes
    axs[0].set_ylabel('x(m)')
    axs[1].set_ylabel('y(m)')
    axs[2].set_ylabel('z(m)')
    axs[2].set_xlabel('time (200 Hz)')

    #plot 3x1 subplot of ee_rx, ee_ry, ee_rz
    fig, axs = plt.subplots(3, 1)
    fig.suptitle('Recorded EE trajectory RPY')

     #apply same scale for all 3 subplots
    # axs[0].set_ylim(-184, -176)
    # axs[1].set_ylim(-5.5, 5.5)
    # axs[2].set_ylim(-85, -75)



    axs[0].plot(t, euler_t[:,0], label='rx')
    axs[1].plot(t, euler_t[:,1], label='ry')
    axs[2].plot(t, euler_t[:,2], label='rz')

    #label axes
    axs[0].set_ylabel('rx(deg)')
    axs[1].set_ylabel('ry(deg)')
    axs[2].set_ylabel('rz(deg)')
    axs[2].set_xlabel('time (200 Hz)')


    plt.show()


def plot_xt():
    load_file_x = '/home/iam-lab/audio_localization/vibrotactile_localization/data/franka_2D_localization/recorded_x_trajectory.npy'
    load_file_xdot = '/home/iam-lab/audio_localization/vibrotactile_localization/data/franka_2D_localization/recorded_xdot_trajectory.npy'
    load_file_x_des = '/home/iam-lab/audio_localization/vibrotactile_localization/data/franka_2D_localization/recorded_x_des_trajectory.npy'
    load_file_xdot_des = '/home/iam-lab/audio_localization/vibrotactile_localization/data/franka_2D_localization/recorded_xdot_des_trajectory.npy'

    x_t = np.load(load_file_x)
    xdot_t = np.load(load_file_xdot)
    x_des_t = np.load(load_file_x_des)
    xdot_des_t = np.load(load_file_xdot_des)

    print(f"size of recorded_trajectory: {len(x_t)}, {len(xdot_t)}") #--> 10, 10


    #plot 2x1 subplots of xt[0] vs time, xdot_t[0] vs time
    fig, axs = plt.subplots(2, 1)
    fig.suptitle('Recorded trajectory for single tap')
    # axs[0].plot(x_t_list[0][:,0], label='x')
    axs[0].plot(x_t[0][:,1], label='y')
    axs[0].plot(x_des_t[0][:,1], label='y_des')
    # axs[0].plot(x_t_list[0][:,2], label='z')

    # axs[1].plot(xdot_t_list[0][:,0], label='x')
    axs[1].plot(xdot_t[0][:,1], label='y')
    axs[1].plot(xdot_des_t[0][:,1], label='y_des')
    # axs[1].plot(xdot_t_list[0][:,2], label='z')

    #label axes
    axs[0].set_ylabel('position (m)')
    axs[1].set_ylabel('velocity (m/s)')
    axs[1].set_xlabel('time (200 Hz)')

    #legend
    axs[0].legend()
    axs[1].legend()

    plt.show()

def plot_xdot_t_all_trials():
    load_file_xdot = '/home/iam-lab/audio_localization/vibrotactile_localization/data/franka_2D_localization/recorded_xdot_trajectory.npy'
    xdot_t = np.load(load_file_xdot)

    trials = len(xdot_t)

    #plot y values for all trials. Make each plot transparent so overlap can be seen
    for i in range(trials):
        plt.plot(xdot_t[i][:,1], label=f'trial {i}', alpha=0.5)

    plt.ylabel('velocity (m/s)')
    plt.xlabel('time (200 Hz)')

    #title
    plt.title('EE velocity_y for all trials')

    plt.show()


def plot_qt():
    load_q_file = '/home/iam-lab/audio_localization/vibrotactile_localization/data/franka_2D_localization/recorded_q_trajectory.npy'
    recorded_q_trajectory = np.load(load_q_file, allow_pickle=True)

    #[q_t, qd_t] for each trial
    print(f"len of recorded_q_trajectory: {len(recorded_q_trajectory)}")

    #print shape of q_t, qd_t
    print(f"shapes: {recorded_q_trajectory[0][0].shape, recorded_q_trajectory[0][1].shape}") #--> ((32990, 7), (32990, 7)) ??? why not 40,000 since 200hz x 2 sec x 10 tirals


# plot_xt()
# plot_xdot_t_all_trials()
plot_entire_state()
