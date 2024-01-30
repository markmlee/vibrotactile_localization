import argparse
from frankapy import FrankaArm
import numpy as np
import sys
import signal


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


    def go_to_init_pose(self):
        print(f"starting motion")
        self.franka.reset_pose()

        # #go to init pose where j6 is 90 degrees
        init_joints = [0,  -8.44859e-01,  0, -2.431180e+00,  0,  3.14159e+00, 7.84695e-01]
        self.franka.goto_joints(init_joints)

    def tap_stick(self, distanceY):
        #move in -Y
        current_pose = self.franka.get_pose()
        current_pose.translation += np.array([0, distanceY, 0]) #-0.01
        self.franka.goto_pose(current_pose, ignore_virtual_walls=True, block=False)

    def move_away_from_stick(self, distanceY):
        #move in +Y
        current_pose = self.franka.get_pose()
        current_pose.translation += np.array([0, distanceY, 0])
        self.franka.goto_pose(current_pose, ignore_virtual_walls=True)

    def execute_motion(self):
        

        # #rotate joint6 to -180 degrees
        # joints = self.franka.get_joints()
        # joints[6] += np.deg2rad(-180)
        # self.franka.goto_joints(joints, ignore_virtual_walls=True)

        sys.exit()


        #go where just about to make contact
        current_pose = self.franka.get_pose()
        current_pose.translation += np.array([0.04, -0.03, -0.1])
        self.franka.goto_pose(current_pose, ignore_virtual_walls=True)

        
        
        #move in +X to get to ring of microphone
        current_pose = self.franka.get_pose()
        current_pose.translation += np.array([+0.07, 0, 0])
        self.franka.goto_pose(current_pose, ignore_virtual_walls=True)

        
        # loop through tapping points for all degrees and distances
        for trials in range(1):
            
            for rotations in range(4):

                print(f" rotation: {rotations} / 4")


                #move in -Y
                current_pose = self.franka.get_pose()
                current_pose.translation += np.array([0, -0.01, 0])
                self.franka.goto_pose(current_pose, ignore_virtual_walls=True )

                #record audio data

                #move in +Y
                current_pose = self.franka.get_pose()
                current_pose.translation += np.array([0, 0.01, 0])
                self.franka.goto_pose(current_pose, ignore_virtual_walls=True)

                
                # rotate 90 degrees
                joints = self.franka.get_joints()
                joints[6] += np.deg2rad(90)
                self.franka.goto_joints(joints, ignore_virtual_walls=True)

        print(f" pulling away")
        #pull back out to avoid collision
        current_pose = self.franka.get_pose()
        current_pose.translation += np.array([-0.1, 0.1, 0])
        self.franka.goto_pose(current_pose, ignore_virtual_walls=True)
        
        print(f"---------- motion complete ----------")
        self.franka.reset_pose()
               
        

# main
# def main():
#     print(" ----- starting script ----- ")
#     franka_robot = FrankaMotion()
#     franka_robot.execute_motion()



# if __name__ == "__main__":
#     main()