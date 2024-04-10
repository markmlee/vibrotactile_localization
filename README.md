# audio_datacollection

## setup
pip install librosa
pip install sounddevice
pip install pysoundfile
pip install hydra-core --upgrade
pip install easydict
pip install wandb

## initial setup
See what mic devices are available. Running code directory `/home/iam-lab/audio_localization/audio_datacollection/scripts`
Change absolute path in datasets.py line 23.

```
python3 record_all_mics.py -l
```

```
9 USB Lavalier Mic Pro: Audio (hw:2,0), ALSA (1 in, 2 out)
10 USB Lavalier Mic Pro: Audio (hw:3,0), ALSA (1 in, 2 out)
11 USB Lavalier Mic Pro: Audio (hw:4,0), ALSA (1 in, 2 out)
12 USB Lavalier Mic Pro: Audio (hw:5,0), ALSA (1 in, 2 out)
13 USB Lavalier Mic Pro: Audio (hw:6,0), ALSA (1 in, 2 out)
14 USB Lavalier Mic Pro: Audio (hw:7,0), ALSA (1 in, 2 out)
```



## running audio with human prelim testing
Setup USB lines, check devicelist and specify devicelist input in the code. 
```
python3 record_with_human.py --record 1
```

To plot
python3 record_with_human.py --record 0



## running audio with franka
Setup USB lines, check devicelist and specify in the code.

To verify all mics are up and registered in the expected sequence, let's record a sequence of taps along the mics and visualize that the expected sequence of mics are responding. Plot from audio .wav files.

```
python3 record_all_mics.py -dlist 9 10 11 12 13 14
```

Visualize plot of .wav files
```
python3 plot_all_mics.py
```

Once verified the mics are well sequenced, let's boot up the franka robot (instructions not specified here) and run the script for data collection. Make sure distance_sample_count, radian_sample_count, total_repeat_count are set to the desired values, as this would dictate how many trial samples are generated. For instance, 50 distance (0.4cm res), 31 (10 degree res), 3 (repeat samples) would result to 50x31x3 = 4650 taps. Sequence would be [50 taps for 31 radians] total of 3 times. (0-1550 sample should repeat 3 times).
```
python3 record_with_franka.py 
```

To record robot ee_pose, run in parallel of the recording script which subscribes to ros topics and saves npy file
```
python3 util_datalogging.py 
```

## training from the collected dataset

Move the dataset into specified directory. Use a GPU-installed desktop with pytorch (conda environment: pytorch3).
Update the train.yaml for 1D line evaluation (corresponding dataset: franka_init_test_6mic), and train2D.yaml for 2D line evaluation (corresponding dataset: franka_2D_localization). 

```
python train.py 
```

## evaluating the trained model
Update the eval.yaml or eval2D.yaml. Specify the desired model path.  
```
python eval.py 
```