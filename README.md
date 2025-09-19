
# *SonicBoom:* : Contact Localization Using Array of Microphones

SonicBoom is a holistic hardware and learning pipeline that enables contact localization through an array of contact microphones. While conventional sound source localization methods effectively triangulate sources in air, localization through solid media with irregular geometry and structure presents challenges that are difficult to model analytically. By leveraging relative features between microphones, SonicBoom achieves localization errors of 0.43cm for in-distribution interactions and maintains robust performance of 2.22cm error even with novel objects and contact conditions. 

Visit the [project website](https://iamlab-cmu.github.io/sonicboom/) for more information and videos.
Our paper can be found [here](https://arxiv.org/abs/2412.09878). 

## Code Description
`/learning` folder contains relevant code to create the model architecture, setup the dataloader, and train the network. 
`/scripts` folder contains relevant code to interface with the robot to collect data, as well as util functions for the microphone processing.

`train2D_UMC.py` is the main training script where UMC1820 is the DAQ model number used for the paper. Previous training scripts utilized separate USB DAQ for audio interface.  


# Audio Data Collection

## Setup
pip install librosa
pip install sounddevice
pip install pysoundfile
pip install hydra-core --upgrade
pip install easydict
pip install wandb
pip install noisereduce
pip install hydra-core --upgrade

## Initial setup
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



## Running audio with human preliminary testing
Setup USB lines, check devicelist and specify devicelist input in the code. 
```
python3 record_with_human.py --record 1
```

To plot
python3 record_with_human.py --record 0



## Robot motion for collecting audio with Franka robot
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

## Training from the collected dataset

Move the dataset into specified directory. Use a GPU-installed desktop with pytorch (conda environment: pytorch3).
Update the train.yaml for 1D line evaluation (corresponding dataset: franka_init_test_6mic), and train2D.yaml for 2D line evaluation (corresponding dataset: franka_2D_localization). 

```
python train.py 
```

## Evaluating the trained model
Update the eval.yaml or eval2D.yaml. Specify the desired model path.  
```
python eval.py 
```


## Inferencing on the robot
Stream the microphones via ROS topics. Input the device IDs according to the number of mics.
```
python sounddevice_ros_publisher_multi.py -d 2 10 11 12 13 14
```

Run the ROS node to monitor audio topic and output contact classification
```
python AudioMonitor.py
```

Visualizing the robot in RViz
```
cd /home/iam-lab/Documents/frankapy/launch
roslaunch franka_cylinder_rviz.launch
```

publish obstacle world
```
cd /home/iam-lab/audio_localization/vibrotactile_localization/scripts/robot_scripts
python publish_world_obstacle.py
```

## Analyzing the time-shift & amplitude vs location data
Uncomment the desired function load_data_for_amplitude() or load_data_for_timeshift() 
```
python process_data.py
```

## Combine datasets
Update the desired directory name

```
python process_gt_label.py
```