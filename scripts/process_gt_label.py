import sys
import os
import numpy as np
import time

# dataset_directory = '/home/mark/audio_learning_project/data/franka_2D_localization_full_UMC_ranged'
# dataset_directory = '/home/mark/audio_learning_project/data/franka_UMC_fixed'
# dataset_directory_angled = '/home/mark/audio_learning_project/data/franka_angled_UMC_full'

# dataset_directory1 = '/home/mark/audio_learning_project/data/wood_T12_L42_Horizontal'
# dataset_directory2 = '/home/mark/audio_learning_project/data/wood_T32_L42_Horizontal'
# dataset_directory3 = '/home/mark/audio_learning_project/data/wood_T22_L42_Horizontal'
# dataset_directory4 = '/home/mark/audio_learning_project/data/wood_T22_L80_Horizontal'

dataset_directory_combined = '/home/mark/audio_learning_project/data/wood_T12_T22_T32_L42_T22_L80_Horizontal_combined'
dataset_directory_new = '/home/mark/audio_learning_project/data/wood_T25_L42_Horizontal'

def get_trials_from_directory(directory):
    """
    input: directory path
    return all trials in a list excluding other files
    """
    #get all directory path to trials
    trial_dir = sorted([os.path.join(directory, f) for f in os.listdir(directory)])

    #filter out directory that does not start with 'trial'
    trial_dir = [d for d in trial_dir if d.split('/')[-1].startswith('trial')]


    
    return trial_dir

def modify_gt_label(directory):
    """
    input: directory path
    offset the gt_label
    """
    #get all directory path to trials
    trial_dir = sorted([os.path.join(directory, f) for f in os.listdir(directory)])

    #filter out directory that does not start with 'trial'
    trial_dir = [d for d in trial_dir if d.split('/')[-1].startswith('trial')]

    len_data = len(trial_dir)

    for trial_n in range(len_data):
        gt_filename = f"{trial_dir[trial_n]}/gt_label.npy"
        if not os.path.exists(gt_filename):
            print(f"Error: {gt_filename} does not exist")
        else:
            gt_label_original = np.load(gt_filename)

            #copy the original gt_label
            gt_label_new = gt_label_original.copy()
            gt_label_new[0] = gt_label_original[0] - 0.1015
            
            # print(f"modified original to new gt_label {gt_label_original} to {gt_label_new}")
            np.save(f"{trial_dir[trial_n]}/gt_label.npy", gt_label_new)

    
    print(f"Completed modifying gt_label for {directory}")
    sys.exit()

def modify_gt_label_angled(directory):
    """
    input: directory path that contains trials to modify and the new GT label
    offset the gt_label
    """
    
    #load new gt_label
    new_gt_filename = f"{directory}/angled_full_label.npy"
    new_gt_label = np.load(new_gt_filename, allow_pickle=True)

    len_new_gt_label = len(new_gt_label)
    print(f"len of new_gt_label {len_new_gt_label}")

    #iterate through each trial and modify the gt_label
    for trial_n in range(len_new_gt_label):
        #get new gt_label (from collision checker [range 0 to 0.203])
        new_gt_label_trial = new_gt_label[trial_n]

        #handle case where new_gt_label contains None
        if new_gt_label_trial[0] is None or new_gt_label_trial[1] is None:
            print(f"new_gt_label_trial {new_gt_label_trial} is None")
            #replace with previous gt_label
            new_gt_label_trial = new_gt_label[trial_n-1].copy()

        new_gt_label_trial[0] = new_gt_label_trial[0] - 0.1015 #offset the gt_label
        # print(f"new_gt_label_trial {new_gt_label_trial}")

        #get old gt_label (from FK with error [range -0.1 to 0.1])
        #folder name
        trial_dir = f"{directory}/trial{trial_n}"
        gt_filename = f"{trial_dir}/gt_label.npy"
        gt_label_original = np.load(gt_filename, allow_pickle=True)

        #modify the gt_label
        gt_label_new = gt_label_original.copy()
        gt_label_new[0] = new_gt_label_trial[0]


        #save the new gt_label
        # print(f"modified original {gt_label_original} to new {gt_label_new}")
        np.save(f"{trial_dir}/gt_label.npy", gt_label_new)

    print(f"Completed modifying gt_label for {directory}")
    sys.exit()

def combine_dataset(combine_dataset_directory_list, combined_dataset_name):
    """
    input: list of dataset directories to combine
    output: save the combined dataset
    
    iterate through all dataset directories and then combine into a single one (should contain wav files and gt_label)
    [trial0-3750, trial0-3750, trial0-479] should be combined into [trial0-7980]
    """

    total_trial_count = 0
    trial_count_list = []
    trial_index_count = 0

    for dataset_directory_n in combine_dataset_directory_list:

        #get all directory path to trials
        trial_dir = sorted([os.path.join(dataset_directory_n, f) for f in os.listdir(dataset_directory_n)])
        #filter out directory that does not start with 'trial'
        trial_dir = [d for d in trial_dir if d.split('/')[-1].startswith('trial')]
        len_data = len(trial_dir)
        total_trial_count += len_data
        trial_count_list.append(len_data)

    print(f"Total number of trials in combined dataset: {total_trial_count}")
    
    #iterate through datasets to copy over to combined dataset
    for idx,dataset_directory_n in enumerate(combine_dataset_directory_list):
        
        print(f"*************going through {dataset_directory_n} *************")
        #get all directory path to trials
        for trial_n in range(trial_count_list[idx]):
            
            #load trials
            trial_dir = f"{dataset_directory_n}/trial{trial_n}"

            #every 100 trial, print out the trial dir for sanity check
            if trial_n % 100 == 0:
                print(f"trial_n {trial_n} trial_dir {trial_dir}")
            
            #copy directory to new combined dataset
            new_trial_dir = f"{combined_dataset_name}/trial{trial_index_count}"
            
            # print(f"copying over {trial_dir} to {new_trial_dir}")
            os.system(f"cp -r {trial_dir} {new_trial_dir}")

            #increment the trial number
            trial_index_count+= 1

    print(f"Completed combining {trial_index_count} files into datasets  {combined_dataset_name}")
    sys.exit()



def main():

    # ----------------------------- ONLY RUN ONCE WHEN MODIFYING GT_LABEL TO OFFSET -----------------------------
    # modify_gt_label(dataset_directory)
    # modify_gt_label_angled(dataset_directory_angled)
    # --------------------------------------------------------------------------------------------------------------------

    # ----------------------------- ONLY RUN ONCE WHEN COMBINING DATASETS -----------------------------

    #iterate through all dataset directories and then combine into a single one
    combine_dataset_directory_list = [dataset_directory_combined, dataset_directory_new]
    # combine_dataset_directory_list = [dataset_directory1, dataset_directory2, dataset_directory3, dataset_directory4]

    combined_dataset_name = '/home/mark/audio_learning_project/data/wood_T12_T22_T25_T32_L42_T22_L80_Horizontal_combined'

    #make the combined dataset directory
    os.system(f"mkdir -p {combined_dataset_name}")
    
    combine_dataset(combine_dataset_directory_list, combined_dataset_name)

    sys.exit()
    # --------------------------------------------------------------------------------------------------------------------
    
    

    trial_dir = get_trials_from_directory(dataset_directory)
    #load data (for multiple mics in device list, get wav files)
    len_data = len(trial_dir)
    print('Total number of trials: ', len_data)

    for trial_n in range(len_data):
        pass
        gt_filename = f"{trial_dir[trial_n]}/gt_label.npy"
        if not os.path.exists(gt_filename):
            print(f"Error: {gt_filename} does not exist")
        else:
            gt_label = np.load(gt_filename)
            print(f"gt_label {gt_label}")
        


if __name__ == '__main__':
    main()