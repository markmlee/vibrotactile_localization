import microphone
import microphone_utils
# from franka_motion import FrankaMotion
import argparse

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--record', type=int, default=0, help='record new? 1/0')
args = parser.parse_args()

def record(devicelist, fs, channels_in, save_path_data, trial_number):
    for i in range(trial_number):
        print(f"trial number: {i}")
        # #record
        trial_count = i
        mic = microphone.Microphone(devicelist, fs, channels_in)
        mic.record_all_mics(save_path=save_path_data, duration=1, trial_count=trial_count)
    

    

def main():
    print(f" ------ starting script ------  ")

    #create instance of microphone class
    devicelist=[9,10]
    number_of_mics = len(devicelist)
    fs = 44100
    channels_in = 1
    save_path_data = "/home/iam-lab/audio_localization/audio_datacollection/data/"

    total_trial_count = 10
    #record
    if args.record == 1:
        record(devicelist, fs, channels_in, save_path_data, total_trial_count)

    

    mics_to_plot = [0,1] #index 01,2,3,4,5,6 -> mic 2,10,11,12,13,14

    #plot recorded wav files
    # microphone_utils.plot_specific_wav_files(devicelist, mics_to_plot, save_path_data, total_trial_count)

    #plot recorded wav files with envelope
    frame_size = 128
    hop_length = 64
    microphone_utils.plot_envelope(frame_size, hop_length, devicelist, mics_to_plot, save_path_data, total_trial_count)



if __name__ == '__main__':

    

    
    
    main()