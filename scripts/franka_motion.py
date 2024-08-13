import argparse
from frankapy import FrankaArm
import numpy as np
import sys
import signal
import time
from autolab_core import RigidTransform

#ros
import rospy
import tf
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

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
        # init_joints = [1.4214274797776991, -0.5763641740554059, -0.043137661899913825, -2.3137106815621444, -0.04326450534330474, 1.7162178359561493, 0.40052493558290536] # table top config 
        init_joints = [1.2613829065054611, -0.8972770578040096, -0.0474416776405632, -2.3463984830755953, -0.07899563970830706, 1.434111323976203, -1.7109319163883936]
        self.franka.goto_joints(init_joints, duration=10, ignore_virtual_walls=True)
 
    def go_to_init_pose_opposite(self):
        print(f"go to reset pose")
        self.franka.reset_joints(duration=10)
    
        init_joints = [1.5278832804517326, -0.784425808546836, 0.2584350609981322, -2.004073806227294, 0.19481513745590165, 1.2958232133057626, 0.8416399667163692]
        self.franka.goto_joints(init_joints, duration=10, ignore_virtual_walls=True)

    def go_to_init_recording_pose(self, x_pos = 0.10, y_pos = 0.35, duration=10):
        print(f"go to init recording pose")

        #set goal pose to be [T: 0.475, 0.05, 0.7], [R: 0,0.707,0,0.707] 
        init_record_pose = RigidTransform(
            translation=np.array([x_pos, y_pos, 0.57]), #0.05 offset added for safety when restoring this init position
            rotation= RigidTransform.rotation_from_quaternion([0,0.707,707,0]),
            from_frame='franka_tool', to_frame='world')

        self.franka.goto_pose(init_record_pose, duration=duration, use_impedance=False, ignore_virtual_walls=True)



    def tap_stick(self, distanceY, duration):
        #move in -Y
        current_pose = self.franka.get_pose()
        current_pose.translation += np.array([0, distanceY, 0]) #-0.01
        self.franka.goto_pose(current_pose, use_impedance=False,duration=duration, ignore_virtual_walls=True, block=False)

    def tap_stick_joint(self, duration, goal_j1_angle=3.0):
        #move j1 to tap stick in x direction
        joints = self.franka.get_joints()
        joints[0] += np.deg2rad(goal_j1_angle)
        self.franka.goto_joints(joints, duration=duration, use_impedance=False, ignore_virtual_walls=True, block=False, joint_impedances = [1000, 1000, 1000, 4000, 1000, 1000, 4000])

    def tap_stick_y_joint(self, duration, goal_j6_angle=3.0):
        #move j6 to tap stick in y direction
        joints = self.franka.get_joints()
        joints[5] += np.deg2rad(goal_j6_angle)
        self.franka.goto_joints(joints, duration=duration, use_impedance=False, ignore_virtual_walls=True, block=False, joint_impedances = [4000, 4000, 4000, 4000, 4000, 1000, 4000])


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

    def move_delta_position(self, x=0.0, y=0.0, z=0.0, duration=5):
        current_pose = self.franka.get_pose()
        current_pose.translation += np.array([x, y, z])
        self.franka.goto_pose(current_pose, duration=duration, use_impedance=False, ignore_virtual_walls=True)

    def move_absolute_position(self, x=0.0, y=0.35, z=0.45, duration=5):
        current_pose = self.franka.get_pose()
        current_pose.translation = np.array([x, y, z])
        self.franka.goto_pose(current_pose, duration=duration, use_impedance=False, ignore_virtual_walls=True)


    def move_with_fixed_orientation(self,x=0.10,y=0.35, z=0.45, current_ee_RigidTransform_rotm=RigidTransform.rotation_from_quaternion([0,0.707,0.707,0]), duration=5): 

        new_record_pose = RigidTransform(
            translation=np.array([x, y, z]),
            rotation= current_ee_RigidTransform_rotm,
            from_frame='franka_tool', to_frame='world')
        self.franka.goto_pose(new_record_pose, duration= duration, use_impedance=False, ignore_virtual_walls=True)


       

    def rotate_j7(self, j7_radian):
        joints = self.franka.get_joints()
        joints[6] = j7_radian
        self.franka.goto_joints(joints, duration=10, ignore_virtual_walls=True)

    def rotate_ee_orientation(self, tap_angle=20):
        """
        rotate the end effector orientation in x,y axis fixed frame w.r.t world frame
        randomly select a rotation angle in x,y axis
        """

        # Get the current pose of the end effector in the world frame
        T_ee_world = self.franka.get_pose()

        # Create a rotation matrix for a 20 degree rotation around the x-axis
        R_rot = RigidTransform.y_axis_rotation(np.deg2rad(tap_angle))

        # Apply the rotation to the current pose
        T_ee_world.rotation = np.dot(R_rot, T_ee_world.rotation)

        # Move the end effector to the new pose
        self.franka.goto_pose(T_ee_world, duration=5, use_impedance=False, ignore_virtual_walls=True)


    def verify_motion_rotate_ee_orientation(self):
        """
        for N times, rotate j7 by 300/N degrees, rotate ee, tap stick, move away from stick, repeat
        """
        N = 4
        j7_radian_list = np.linspace(-2.7, 2.7, N)

        for i in range(N):
            print(f"A) rotating j7 to {j7_radian_list[i]} radians, in degrees: {np.rad2deg(j7_radian_list[i])}")
            self.rotate_j7(j7_radian_list[i])

            # store joint position
            joints_before_contact = self.get_joints() 
            
            #random tap angle degree from [0,20]
            tap_angle = np.random.uniform(0, 20)
            print(f"B) rotating ee orientation X by {tap_angle} degrees")
            self.rotate_ee_orientation(tap_angle)

            print(f"C) tapping stick")
            goal_j1_angle = 5.0
            record_duration = 2
            self.tap_stick_joint(duration=record_duration/2, goal_j1_angle = goal_j1_angle)

            time.sleep(3)

            #go to stored joint position
            print(f"D) moving away from stick")
            self.go_to_joint_position(joints_before_contact, duration=2)

            time.sleep(3)

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

    def save_prediction(self, height, x,y , directory_path, trial_count):
        """
        save predicted contact point
        """
        
        save_folder_path = f"{directory_path}trial{trial_count}/"
        os.makedirs(save_folder_path, exist_ok=True)

        rad = np.arctan2(y, x)
        if rad < 0:
            rad += 2*np.pi


        np.save(f"{save_folder_path}predicted_output_HeightRad.npy", [height, rad])
        
    def xy_to_radians(self, x, y):
        """
        Convert x,y into radians from 0 to 2pi
        """
        rad = np.arctan2(y, x)
        if rad < 0:
            rad += 2*np.pi

        return rad
    
    def radians_to_xy_on_cylinder(self, rad, cylinder_radius):
        """
        Convert radians to x,y with radius included
        """
        x = np.cos(rad) * cylinder_radius
        y = np.sin(rad) * cylinder_radius

        return x, y
    
    def transform_origin_to_cylinder(self, rad_input, cylinder_transform_offset):
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

    
    def transform_predicted_XYZ_to_EE_XYZ(self, x,y,z, cylinder_radius, cylinder_transform_offset):
        """
        Transform the predicted contact pt XYZ (based on dataset cylinder frame) to the EE XYZ (to visualize on RVIZ on EE frame)
        1. post process XY to radian back to XY
        2. align the origin of the cylinder frame to the EE origin
        3. convert to Point Msg
        """

        #convert xy into radians, then project back to x,y with radius mult
        radians = self.xy_to_radians(x, y)

        #transform origin to cylinder EE origin
        radians = self.transform_origin_to_cylinder(radians, cylinder_transform_offset)
        x_on_cylinder, y_on_cylinder = self.radians_to_xy_on_cylinder(radians, cylinder_radius)


        transformed_point = Point()
        transformed_point.x = x_on_cylinder
        transformed_point.y = y_on_cylinder
        transformed_point.z = (-1*z / 100)  #origin opposite from dataset and RViz. convert cm to m
        transformed_point.z += 0.0 #add fine tuning offset

        return transformed_point
    
    
    def create_marker(self, id, contact_pt, scale_xyz_list, color_argb_list, lifetime=2):
        # Create a marker
        """
        input: Point XYZ rosmsg
        """
        marker = Marker()
        marker.header.frame_id = "cylinder_origin"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "contact_point"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = contact_pt.x
        marker.pose.position.y = contact_pt.y
        marker.pose.position.z = contact_pt.z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale_xyz_list[0]
        marker.scale.y =  scale_xyz_list[1]
        marker.scale.z =  scale_xyz_list[2]
        marker.color.a = color_argb_list[0] 
        marker.color.r = color_argb_list[1]
        marker.color.g = color_argb_list[2] 
        marker.color.b = color_argb_list[3] 

        marker.lifetime = rospy.Duration(lifetime)

        return marker


        




# main
def main():
    print(" ----- starting script ----- ")
    franka_robot = FrankaMotion()
    franka_robot.go_to_init_pose()
    # franka_robot.execute_motion()



if __name__ == "__main__":
    main()