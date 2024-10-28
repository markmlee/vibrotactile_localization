import sys
import os
import re
import numpy as np
import time
import matplotlib.pyplot as plt

directory = '/home/mark/audio_learning_project/data/wood_horizontal_opposite_verticalv3'


def rename_wav9_to_wav2(trial_dir):
    """
    Rename the wav files from mic9 to mic2
    """

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


def rename_trial_directories_from_0_to_len_data(trial_dirs):
    """
    When there are removed trials, rename the trial directories from 0 to len_data
    """

    # Rename trial directories from 0 to len_data
    for new_index, current_trial_dir in enumerate(trial_dirs):
        new_trial_dir = f"{directory}/trial{new_index}"

        if os.path.exists(current_trial_dir) and current_trial_dir != new_trial_dir:
            # Uncomment the next line to actually perform the renaming
            os.rename(current_trial_dir, new_trial_dir)
            print(f"Renamed {current_trial_dir} to {new_trial_dir}")

    

def natural_sort_key(s):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

def main():
    # Get all directory paths to trials
    trial_dirs = [os.path.join(directory, f) for f in os.listdir(directory)]

    # Filter out directories that do not start with 'trial'
    trial_dirs = [d for d in trial_dirs if os.path.basename(d).startswith('trial')]

    # Sort the trial directories using natural sorting
    trial_dirs.sort(key=lambda x: natural_sort_key(os.path.basename(x)))


    # ----------------- Rename wav2 to wav9 ------------------------------------------
    rename_wav9_to_wav2(trial_dirs)
    # --------------------------------------------------------------------------------

    # ----------------- Rename trial directories from 0 to len_data -------------------
    # rename_trial_directories_from_0_to_len_data(trial_dirs)


if __name__ == '__main__':
    main()