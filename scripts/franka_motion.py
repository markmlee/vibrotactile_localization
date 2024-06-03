import argparse
from frankapy import FrankaArm
import numpy as np
import sys
import signal
import time
from autolab_core import RigidTransform
import rospy

import os
# class for motion
class FrankaMotion:
    def __init__(self):
        print(f"initializing franka")
        self.franka = FrankaArm(with_gripper=False)
    #     signal.signal(signal.SIGINT, self.signal_handler)
        
    # def signal_handler(self,sig, frame):
    #     print('You pressed Ctrl+C!')
    #     self.franka.reset_pose()
    #     sys.exit(0)

    def reset_joints(self):
        self.franka.reset_joints(duration=10)

    def go_to_init_pose(self):
        print(f"go to reset pose")
        self.franka.reset_joints(duration=10)

        # #go to init pose where j6 is 90 degrees
        # init_joints = [0,  -8.44859e-01,  0, -2.431180e+00,  0,  3.14159e+00, 7.84695e-01] #j6 = 90 degrees from reset pose
        # init_joints = [-0.6140042477755041, -0.9133502268939996, 0.5870043861405891, -2.3453908448474925, 1.9926752161905008, 2.9358218419186985, -0.6770297919229847]
        init_joints = [1.4214274797776991, -0.5763641740554059, -0.043137661899913825, -2.3137106815621444, -0.04326450534330474, 1.7162178359561493, 0.40052493558290536] # table top config 
        self.franka.goto_joints(init_joints, duration=10, ignore_virtual_walls=True)
 

    def go_to_init_recording_pose(self, duration=10):
        print(f"go to init recording pose")

        #set goal pose to be [T: 0.475, 0.05, 0.7], [R: 0,0.707,0,0.707] 
        init_record_pose = RigidTransform(
            translation=np.array([0.10, 0.35, 0.45]), #0.05 offset added for safety when restoring this init position
            rotation= RigidTransform.rotation_from_quaternion([0,0.707,707,0]),
            from_frame='franka_tool', to_frame='world')


        self.franka.goto_pose(init_record_pose, duration=10, use_impedance=False, ignore_virtual_walls=True)

    def tap_stick(self, distanceY, duration):
        #move in -Y
        current_pose = self.franka.get_pose()
        current_pose.translation += np.array([0, distanceY, 0]) #-0.01
        self.franka.goto_pose(current_pose, use_impedance=False,duration=duration, ignore_virtual_walls=True, block=False)

    def tap_stick_joint(self, duration, goal_j1_angle=3.0):
        #move j1
        joints = self.franka.get_joints()
        joints[0] += np.deg2rad(goal_j1_angle)
        self.franka.goto_joints(joints, duration=duration, use_impedance=False, ignore_virtual_walls=True, block=False, joint_impedances = [1000, 4000, 4000, 4000, 4000, 4000, 4000])

    def move_away_from_stick_joint(self, duration):
        #move j1
        joints = self.franka.get_joints()
        joints[0] -= np.deg2rad(3)
        self.franka.goto_joints(joints, duration=duration, use_impedance=False, ignore_virtual_walls=True)

    def move_away_from_stick(self, distanceY):
        #move in +Y
        current_pose = self.franka.get_pose()
        current_pose.translation += np.array([0, distanceY, 0])
        self.franka.goto_pose(current_pose, use_impedance=False, ignore_virtual_walls=True)

    def move_along_pipe(self,x=0.10,y=0.35, z=0.45, current_ee_RigidTransform_rotm=RigidTransform.rotation_from_quaternion([0,0.707,0.707,0])): 

        new_record_pose = RigidTransform(
            translation=np.array([x, y, z]),
            rotation= current_ee_RigidTransform_rotm,
            from_frame='franka_tool', to_frame='world')
        self.franka.goto_pose(new_record_pose, use_impedance=False, ignore_virtual_walls=True)


       

    def rotate_j7(self, j7_radian):
        joints = self.franka.get_joints()
        joints[6] = j7_radian
        self.franka.goto_joints(joints, duration=10, ignore_virtual_walls=True)

    def get_joints(self):
        return self.franka.get_joints()
    
    def go_to_joint_position(self, joints, duration):
        self.franka.goto_joints(joints, ignore_virtual_walls=True, duration=duration)

    def get_ee_pose(self):
        return self.franka.get_pose()

    def get_ee_pose_des(self):
        return self.franka.get_robot_state()['pose_desired']
    
    def get_joints_des(self):
        return self.franka.get_robot_state()['joints_desired']
    
    def get_joint_vel_des(self):
        return self.franka.get_robot_state()['joint_velocities_desired']
    
    
    def get_ee_velocity(self):
        q = self.franka.get_joints()
        q_dot = self.franka.get_joint_velocities()
        jacobian = self.franka.get_jacobian(q)
        ee_velocity = np.dot(jacobian, q_dot)

        #xyz linear velocity
        ee_velocity = ee_velocity[0:3]

        return ee_velocity
    
    def get_ee_velocity_des(self):
        q_des = self.get_joints_des()
        q_dot_des = self.get_joint_vel_des()
        jacobian = self.franka.get_jacobian(q_des)
        ee_velocity_des = np.dot(jacobian, q_dot_des)

        #xyz linear velocity
        ee_velocity_des = ee_velocity_des[0:3]
        return ee_velocity_des
    
    def record_trajectory(self, duration, dt):
        T = duration
        t = np.arange(0, T, dt)
        x_t = np.zeros((len(t), 3))
        xdot_t = np.zeros((len(t), 3))
        x_t_des = np.zeros((len(t), 3))
        xdot_t_des = np.zeros((len(t), 3))
        q_t = np.zeros((len(t), 7))
        q_tau = np.zeros((len(t), 7))

        # print(f"traj: starting to record now for {T} seconds")
        start = time.time()
        rate = rospy.Rate(100)

        for i in range(len(t)):
            x_t[i] = self.get_ee_pose().translation
            xdot_t[i] = self.get_ee_velocity()
            x_t_des[i] = self.get_ee_pose_des().translation
            xdot_t_des[i] = self.get_ee_velocity_des()
            q_t[i] = self.franka.get_joints()
            q_tau[i] = self.franka.get_joint_torques() 
            

            #print ee force (CUTS OUT printing F/T sensor data in middle)
            # print(f"ee force: {self.franka.get_ee_force_torque()}")

            rate.sleep()
        
        # print(f"finished recording in {time.time()-start} seconds")
        self.x_t, self.xdot_t, self.x_t_des, self.xdot_t_des, self.q_t, self.q_tau = x_t, xdot_t, x_t_des, xdot_t_des, q_t, q_tau
        return x_t, xdot_t, x_t_des, xdot_t_des, q_t, q_tau
    
    def get_trajectory_results(self):
        print(f"getting recorded trajectory")
        return self.x_t, self.xdot_t, self.x_t_des, self.xdot_t_des
    
    def save_recorded_trajectory(self, directory_path, trial_count, x_t, xdot_t, x_t_des, xdot_t_des, q_t, q_tau, goal_j1_angle):
        """
        save recorded trajectory
        """

        save_folder_path = f"{directory_path}trial{trial_count}/"
        os.makedirs(save_folder_path, exist_ok=True)

        np.save(f"{save_folder_path}x_t.npy", x_t)
        np.save(f"{save_folder_path}xdot_t.npy", xdot_t)
        np.save(f"{save_folder_path}x_t_des.npy", x_t_des)
        np.save(f"{save_folder_path}xdot_t_des.npy", xdot_t_des)
        np.save(f"{save_folder_path}q_t.npy", q_t)
        np.save(f"{save_folder_path}q_tau.npy", q_tau)
        np.save(f"{save_folder_path}goal_j1_angle.npy", goal_j1_angle)

        # print(f"saved recorded trajectory to {save_folder_path}")

        




# main
def main():
    print(" ----- starting script ----- ")
    franka_robot = FrankaMotion()
    franka_robot.go_to_init_pose()
    # franka_robot.execute_motion()



if __name__ == "__main__":
    main()