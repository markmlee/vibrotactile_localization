import microphone
import microphone_utils

def main():
    print(f" ------ starting script ------  ")

    #create instance of microphone class
    devicelist=[9,10,11,12,13,14]
    number_of_mics = len(devicelist)
    fs = 44100
    channels_in = 1

    mic = microphone.Microphone(devicelist, fs, channels_in)

    save_path_data = "/home/iam-lab/audio_localization/audio_datacollection/data/"

    
    #record
    trial_count = 0
    mic.record_all_mics(save_path=save_path_data, duration=3, trial_count=trial_count)

    #plot
    microphone_utils.plot_wav_files(devicelist, trial_count ,save_path_data)




if __name__ == '__main__':
    main()