import microphone
import microphone_utils
from franka_motion import FrankaMotion
import os
import time

def main():
    print(f" ------ starting script ------  ")

    #create instance of microphone class
    devicelist=[2,10,11,12,13,14]
    number_of_mics = len(devicelist)
    fs = 44100
    channels_in = 1

    #create a folder to save data
    save_path_data = "/home/iam-lab/audio_localization/audio_datacollection/data/franka_init_test_6mic/"
    os.makedirs(save_path_data, exist_ok=True)

    franka_robot = FrankaMotion()
    franka_robot.go_to_init_pose()

    record_duration = 3

    for h in range(3):
        #get ground truth label [distance along cylinder, joint 6]
        gt_label = [0,0]

        #move robot 
        franka_robot.tap_stick(-0.05)
        mic = microphone.Microphone(devicelist, fs, channels_in)
        mic.record_all_mics(save_path=save_path_data, duration=record_duration, trial_count=h, gt_label=gt_label)
        time.sleep(record_duration*2) #wait for mic to finish recording before calling next skill
        franka_robot.move_away_from_stick(0.05)

        


    # #record
    # trial_count = 0
    # mic.record_all_mics(save_path=save_path_data, duration=3, trial_count=trial_count)

    # #plot
    # microphone_utils.plot_wav_files(devicelist, trial_count ,save_path_data)




if __name__ == '__main__':
    main()