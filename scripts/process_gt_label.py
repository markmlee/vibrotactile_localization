import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
# dataset_directory = '/home/mark/audio_learning_project/data/franka_2D_localization_full_UMC_ranged'
# dataset_directory = '/home/mark/audio_learning_project/data/franka_UMC_fixed'
dataset_directory_GT_modify = '/home/mark/audio_learning_project/data/test_generalization/cross_easy_X_15_Left'
# dataset_directory_GT_modify = '/home/mark/audio_learning_project/data/wood_T22_L42_Horizontal_opposite'

#original without suction
# dataset_directory1a = '/home/mark/audio_learning_project/data/wood_T12_L42_Horizontal' #DONT USE BC NOISY
dataset_directory1a = '/home/mark/audio_learning_project/data/wood_T12_L42_Horizontal_cleaned'
dataset_directory1b = '/home/mark/audio_learning_project/data/wood_T22_L42_Horizontal'
dataset_directory1c = '/home/mark/audio_learning_project/data/wood_T25_L42_Horizontal'
dataset_directory1d = '/home/mark/audio_learning_project/data/wood_T32_L42_Horizontal'
dataset_directory1e = '/home/mark/audio_learning_project/data/wood_T22_L80_Horizontal'

#with suction
dataset_directory2a = '/home/mark/audio_learning_project/data/wood_T12_L42_Horizontal_v2'
dataset_directory2b = '/home/mark/audio_learning_project/data/wood_T32_L42_Horizontal_v2'
dataset_directory2c = '/home/mark/audio_learning_project/data/wood_T22_L42_Horizontal_v2'
dataset_directory2d = '/home/mark/audio_learning_project/data/wood_T25_L42_Horizontal_v2'

#with opposite side hits
dataset_directory3a = '/home/mark/audio_learning_project/data/wood_T12_L42_Horizontal_opposite'
dataset_directory3b = '/home/mark/audio_learning_project/data/wood_T22_L42_Horizontal_opposite'


#with vertical side hits
dataset_directory4a = '/home/mark/audio_learning_project/data/wood_T12_L42_Vertical'
dataset_directory4b = '/home/mark/audio_learning_project/data/wood_T22_L42_Vertical'
dataset_directory4c = '/home/mark/audio_learning_project/data/wood_T32_L42_Vertical'
dataset_directory4d = '/home/mark/audio_learning_project/data/wood_T25_L42_Vertical'


# dataset_directory5 = '/home/mark/audio_learning_project/data/wood_T12_T22_T25_T32_L42_T22_L80_Horizontal_combinedv2'

# dataset_directory6 = '/home/mark/audio_learning_project/data/wood_T12_L42_Horizontal_opposite'
# dataset_directory7 = '/home/mark/audio_learning_project/data/wood_T22_L42_Horizontal_opposite'


dataset_directory_combined = '/home/mark/audio_learning_project/data/wood_suctionOnly_horizontal_opposite_verticalv2'
# dataset_directory_combined = '/home/mark/audio_learning_project/data/wood_T12_T22_T32_L42_T22_L80_Horizontal_combined'
# dataset_directory_new = '/home/mark/audio_learning_project/data/wood_T25_L42_Horizontal'

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

def modify_gt_label_height_only(directory):
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




# Function to linearly map input values to output values
def radian_map_opposite_stick_values(input_values):
    # Input and output arrays
    X = np.array([-2.7, -2.1, -1.5, -0.9, -0.3, 0.3, 0.9, 1.5, 2.1, 2.7])
    Y = np.array([0.9, 1.5, 2.1, 2.7, np.pi, -2.7, -2.1, -1.5, -0.9, -0.3])

    return np.interp(input_values, X, Y)

def modify_gt_label_radian_only(directory):
    """
    input: directory path
    offset the gt_label radian for the opposite stick label (offset by mapping value function)
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
            gt_label_new[1] = radian_map_opposite_stick_values(gt_label_original[1]) 
            
            print(f"modified original to new gt_label {gt_label_original} to {gt_label_new}")
            np.save(f"{trial_dir[trial_n]}/gt_label.npy", gt_label_new)

    
    print(f"Completed modifying gt_label for {directory}")
    sys.exit()

def modify_gt_label_height_radian_from_input_label(directory):
    """
    input: directory path that contains trials to modify and the new GT label
    offset the gt_label
    """
    
    #load new gt_label
    new_gt_filename = f"{directory}/label.npy"
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

        #new_gt_label_trial[0] = new_gt_label_trial[0] - 0.1015 #offset the gt_label

        
        #get old gt_label (from FK with error [range -0.1 to 0.1])
        #folder name
        trial_dir = f"{directory}/trial{trial_n}"
        gt_filename = f"{trial_dir}/gt_label.npy"
        gt_label_original = np.load(gt_filename, allow_pickle=True)

        #modify the gt_label with the new offset height and angle from new file
        gt_label_new = gt_label_original.copy()
        gt_label_new[0] = new_gt_label_trial[0]
        gt_label_new[1] = new_gt_label_trial[1]


        #save the new gt_label
        print(f"modified original height: {gt_label_original[0]:.2f}, angle: {gt_label_original[1]:.2f} --> to height: {gt_label_new[0]:.2f}, angle: {gt_label_new[1]:.2f} for trial {trial_n}")
        
        
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

def plot_height_radian(directory):
    """
    go through entire directory and get the height and radian
    2D plot of height and radian 
    """

    #get all directory path to trials
    trial_dir = sorted([os.path.join(directory, f) for f in os.listdir(directory)])

    #filter out directory that does not start with 'trial'
    trial_dir = [d for d in trial_dir if d.split('/')[-1].startswith('trial')]

    len_data = len(trial_dir)

    height_list = []
    radian_list = []

    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    for trial_n in range(len_data):
        gt_filename = f"{trial_dir[trial_n]}/gt_label.npy"
        if not os.path.exists(gt_filename):
            print(f"Error: {gt_filename} does not exist")
        else:
            gt_label = np.load(gt_filename)
            height_list.append(gt_label[0])
            radian_list.append(gt_label[1])

            #every 5 trial, print out the height and radian for sanity check
            if trial_n % 1 == 0:
                #print height and radian up to 2 decimal places
                print(f"height:{gt_label[0]:.2f}, radian: {gt_label[1]:.2f}, trial_n {trial_n}")
                #sleep for 0.1 seconds
                # time.sleep(0.1)

    #plot the height and radian
    plt.scatter(height_list, radian_list)
    plt.xlabel('Height')
    plt.ylabel('Radian')
    plt.title('Height vs Radian')
    plt.show()

    sys.exit()

def combine_delete_trials(delete_trials_list):
    return sorted(set(trial for start, end in delete_trials_list for trial in range(start, end )))

def delete_trials_and_reenumerate(dataset_dir, data_interval_length, delete_trials_list):
    """
    load the directory to the single dataset
    load the interval legnth and the list of trials to remove
    delete the trials from the dataset
    reenumerate the remaining trials
    """

    # given list of delete_trials_list that only contains range [ [start,end],..., [start,end] ]
    # combine into a single list of all trials to delete
    delete_trial_all_list = combine_delete_trials(delete_trials_list)
    print(f"delete_trial_all_list {delete_trial_all_list}")

    #get all directory path to trials
    dir = sorted([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.startswith('trial')], key=lambda x: int(os.path.basename(x)[5:]))                #filter out directory that does not start with 'trial'
    dir = [d for d in dir if d.split('/')[-1].startswith('trial')]

    len_data = len(dir)

    for trial_n in range(len_data):

        if trial_n in delete_trial_all_list:
            #delete the trial
            print(f"deleting {dir[trial_n]}")
            os.system(f"rm -r {dir[trial_n]}")

    
    # #re-enumerate the remaining trials

    #get all directory path to trials
    dir = sorted([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.startswith('trial')], key=lambda x: int(os.path.basename(x)[5:]))                #filter out directory that does not start with 'trial'
    dir = [d for d in dir if d.split('/')[-1].startswith('trial')]

    len_data_trimmed = len(dir)
    print(f"len_data_trimmed {len_data_trimmed}")

    for trial_n in range(len_data_trimmed):
        
        #rename the trial to the new trial number
        new_trial_dir = f"{dataset_dir}trial{trial_n}"
        print(f"renaming {dir[trial_n]} to {new_trial_dir}")
        os.rename(dir[trial_n], new_trial_dir)
        time.sleep(0.05)

  

def main():

    # ----------------- For removing bad GT data observed from collision checker ------------------------------------------
    # dataset_dir = "/home/mark/audio_learning_project/data/wood_T12_L42_Horizontal_v2/"
    # data_interval_length = 5
    # delete_trials_list = [[100,110], [250,260]]#, [320,340], [980,1000], [1460,1480], [1500,1520]] #[100-120], [320-340], [980-1000], [1460-1480], [1500-1520]
    # delete_trials_and_reenumerate(dataset_dir, data_interval_length, delete_trials_list)
    # sys.exit()

    # ----------------------------- ONLY RUN ONCE WHEN MODIFYING GT_LABEL TO OFFSET -----------------------------
    # modify_gt_label_height_only(dataset_directory)
    modify_gt_label_height_radian_from_input_label(dataset_directory_GT_modify) #USE THIS FUNCTION

    # modify_gt_label_radian_only(dataset_directory_GT_modify)
    # --------------------------------------------------------------------------------------------------------------------

    


    # ----------------------------- VISUALIZE THE GT LABELS FOR SANITY CHECK -----------------------------
    # dir_to_visualize = dataset_directory6
    # plot_height_radian(dir_to_visualize)





    # ----------------------------- ONLY RUN ONCE WHEN COMBINING DATASETS -----------------------------

    #iterate through all dataset directories and then combine into a single one
    # combine_dataset_directory_list = [dataset_directory_combined, dataset_directory_new]
    # combine_dataset_directory_list = [dataset_directory1, dataset_directory2, dataset_directory3, dataset_directory4, dataset_directory5, dataset_directory6, dataset_directory7]
    combine_dataset_directory_list = [
        dataset_directory1a, dataset_directory1b, dataset_directory1c, dataset_directory1d, dataset_directory1e, 
        dataset_directory2a, dataset_directory2b, dataset_directory2c, dataset_directory2d, 
        dataset_directory3a, dataset_directory3b,
        dataset_directory4a, dataset_directory4b, dataset_directory4c, dataset_directory4d]
    
    
    # combine_dataset_directory_list = [
    #     dataset_directory2a, dataset_directory2b, dataset_directory2c, dataset_directory2d, 
    #     dataset_directory3a, dataset_directory3b,
    #     dataset_directory4a, dataset_directory4b, dataset_directory4c, dataset_directory4d]
    

    #make the combined dataset directory
    os.system(f"mkdir -p {dataset_directory_combined}")
    
    combine_dataset(combine_dataset_directory_list, dataset_directory_combined)

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