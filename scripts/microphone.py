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

class Microphone:
    def __init__(self, devicelist, fs=44100, channels_in=1):
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

        def callback(data, frames, time, status):
 
            q.put(data.copy())

        return callback


    def init_mic_queue(self):
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
        self.record_duration = duration

    def set_path(self, path):
        self.path = path

    def record_all_mics(self, save_path, duration=1, trial_count=0, gt_label=[0,0]):

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
            print(f"Recording for {duration} seconds...")
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
        




    


    