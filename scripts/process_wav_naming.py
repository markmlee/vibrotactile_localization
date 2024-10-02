import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

directory = '/home/mark/audio_learning_project/data/wood_suctionOnly_horizontal_opposite_verticalv2/'


def main():

    #get all directory path to trials
    trial_dir = sorted([os.path.join(directory, f) for f in os.listdir(directory)])

    #filter out directory that does not start with 'trial'
    trial_dir = [d for d in trial_dir if d.split('/')[-1].startswith('trial')]

    len_data = len(trial_dir)

    for trial_n in range(len_data):
        current_wav_filename = f"{trial_dir[trial_n]}/mic9.wav"
        new_wav_filename = f"{trial_dir[trial_n]}/mic2.wav"
        
        if os.path.exists(current_wav_filename):


            #rename the current file to new file
            os.rename(current_wav_filename, new_wav_filename)

            #print out the current and new filename
            print(f"Renamed {current_wav_filename} to {new_wav_filename}")

        else:
            pass


if __name__ == '__main__':
    main()