#!/usr/bin/env python3
"""
Class for recording audio files from microphones in a background thread.
Input paramter: duration of recording in seconds.
Output: .wav files in specified path directory.
"""
import argparse
import queue
import sys

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from soundfile import SoundFile
from scipy.io.wavfile import read

import time
import wave
import sys
import os
import contextlib
import time

#ros
import rospy
import sounddevice as sd
import soundfile as sf

sys.path.append('/home/iam-lab/audio_localization/catkin_ws/src/sounddevice_ros/msg')
from sounddevice_ros.msg import AudioInfo, AudioData
import numpy  # Make sure NumPy is loaded before it is used in the callback

class Microphone:
    def __init__(self, devicelist, fs=44100, channels_in=1):
        """
        Initialize the Microphone class with the given device list.
        
        Args:
            devicelist: List of device IDs for the microphones to be used.
            fs (int, optional): Sampling frequency. Defaults to 44100 Hz.
            channels_in (int, optional): Number of input channels. Defaults to 1.
        """
        self.record_duration = 1
        self.fs = fs
        self.path = "./"
        self.devicelist = devicelist
        self.number_of_mics = len(devicelist)

        self.queue_list = []
        self.stream_sd_list = []
        self.channels_in = channels_in

        self.init_mic_queue()


    def audio_callback(self, q):
        """
        Creates a callback function for the audio stream.
        
        Args:
            q (queue.Queue): Queue to which audio data will be put.

        Returns:
            function: A callback function that is called by the audio stream.
        """

        def callback(data, frames, time, status):
 
            q.put(data.copy())

        return callback


    def init_mic_queue(self):
        """
        Initializes a queue and an input stream for each microphone device.
        """
        for i in range(self.number_of_mics):
            #create instance of queue for each device
            queue_name = f"q{i}" #q0 = queue.Queue() ... q6 = queue.Queue()
            queue_name = queue.Queue() 
            self.queue_list.append(queue_name)
            
            #create instance of stream for each device
            stream_file = sd.InputStream(
                device=self.devicelist[i], channels=self.channels_in,
                samplerate=self.fs, callback=self.audio_callback(self.queue_list[i]))

            self.stream_sd_list.append(stream_file)


            

    def set_record_duration(self, duration):
        """
        Set the duration for which the microphones will record audio.
        
        Args:
            duration (int): Duration in seconds.
        """
        self.record_duration = duration


    def record_all_mics(self, save_path, duration=1, trial_count=0, gt_label=[0,0]):
        """
        Record audio from all microphones and save it to the specified path.

        Args:
            save_path (str): Path where the audio files will be saved.
            duration (int, optional): Duration in seconds for the recording. Defaults to 1.
            trial_count (int, optional): Trial number for the recording. Defaults to 0.
            gt_label (list, optional): Ground truth label for corrresponding audio files. i.e, used for 2D localization. Defaults to [0, 0].
        """

        #create a folder for each trial
        save_folder_path = f"{save_path}trial{trial_count}/"
        os.makedirs(save_folder_path, exist_ok=True)


        #save ground truth label to folder as npy file
        np.save(f"{save_folder_path}gt_label.npy", gt_label)


        #create soundfile for each device
        SoundFile_list = []

        for i in range(self.number_of_mics):
            #file name with save path
            file_name = f"{save_folder_path}mic{self.devicelist[i]}.wav"
            # print(f"file name: {file_name}")

            #if file exists, delete it and recreate it
            try:
                os.remove(file_name)
            except OSError:
                pass

            #create soundfile for each device
            sf = SoundFile(
                file=file_name,
                mode="x",
                samplerate=int(self.fs),
                channels=self.channels_in,
            )

            SoundFile_list.append(sf)


        # record audio for all 6 streams for set duration
        start_time = time.time()
        with contextlib.ExitStack() as stack:
            for stream in self.stream_sd_list:
                stack.enter_context(stream)
            # print(f"Audio: Recording for {duration} seconds...")
            try:
                while time.time() - start_time < duration:
                    for i in range(self.number_of_mics):
                        SoundFile_list[i].write(self.queue_list[i].get())
            except Exception as e:
                print(f"\nInterrupted due to error: {e}")

        #close all soundfiles
        for i in range(self.number_of_mics):
            SoundFile_list[i].close()

        #stop all contexts
        
    def ros_publish_mics(self, output_topic_list):
        """
        Main blocking function to continuously publish audio data to ROS topics.

        Args:
            output_topic_list (list): List of ROS topics to which the audio data will be published.
        """

        #default output topic list based on number of mics... [audio0, audio1, audio2, audio3, audio4, audio5]
        if len(output_topic_list) == 0:
            output_topic_list = [f"/audio{i}" for i in range(self.number_of_mics)]

        #create ROS publisher for each device
        pub_list = []
        for i in range(self.number_of_mics):
            pub = rospy.Publisher(output_topic_list[i], AudioData, queue_size=10)
            pub_list.append(pub)

        rospy.init_node('audio_pub', anonymous=True)
        
        #rate
        rate = rospy.Rate(100) # 10hz
    
        #publish audio data to each topic
        while not rospy.is_shutdown():
            for i in range(self.number_of_mics):
                audio_data = AudioData()
                audio_data.data = self.queue_list[i].get()
                pub_list[i].publish(audio_data)

            rate.sleep()
        

    



    


    