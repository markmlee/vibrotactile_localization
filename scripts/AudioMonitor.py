#!/usr/bin/env python
import sys
import os
import numpy as np
import threading

#hydra
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig

#torch
import torch
import torchaudio

#ros
import rospy
from std_msgs.msg import String, Int32
from geometry_msgs.msg import Point
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../catkin_ws/sounddevice_ros/msg')))
from sounddevice_ros.msg import AudioInfo, AudioData

#custom models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../learning')))
from models.CNN import CNNRegressor2D

#custom utils
import microphone_utils as mic_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../learning')))
from transforms import to_mel_spectrogram, get_signal

""""
Class for simply monitoring audio data from ROS topics and then outputting processed data.
This class is intended to be used stand-alone as if it's a sensor node.

Author: Mark Lee (MoonRobotics@cmu.edu)
Created: 2024-04-09
"""

TRANSFORMS = {
    'wav' : get_signal,
    'mel': to_mel_spectrogram
}

class AudioMonitor:
    def __init__(self, cfg, audio_topics, fs, buffer_duration, sample_duration, main_loop_rate):
        """
        Initializes the AudioMonitor class with the given audio topics.
        
        Args:
            audio_topics (list of str): List of ROS topic names to subscribe to for audio data.
        """
        self.cfg = cfg
        self.fs = fs  # Sample rate of the audio data
        self.num_channels = 1 # Number of audio channels
        self.sample_duration = sample_duration # Size of the audio window for processing (in seconds)
        self.buffer_duration = buffer_duration # Duration of the rolling buffer to keep last previous samples (in seconds)
        self.num_mics = len(audio_topics) # Number of microphones
        self.main_loop_rate = main_loop_rate # Rate of the main loop to process audio data
        self.audio_transform = TRANSFORMS[cfg.audio_transform] # Transform to apply to the audio data


        #dictionary to store rolling buffers for each audio topic
        self.rolling_buffers = {topic: np.zeros((1, self.num_channels)) for topic in audio_topics}
        self.buffer_sizes = np.zeros(self.num_mics)

        # lock to prevent race condition in the callback
        self.lock = threading.Lock()

        self.audio_subscribers = []
        for topic in audio_topics:
            sub = rospy.Subscriber(topic, AudioData, lambda msg, topic=topic: self.audio_callback(msg, topic), queue_size=10)
            self.audio_subscribers.append(sub)

        #publishers
        self.pub_contactevent = rospy.Publisher('/contact_event', Int32, queue_size=10) #--> 0: no contact, 1: contact

        # ROS publisher for the contact location
        self.pub_contactloc = rospy.Publisher('/contact_location', Point, queue_size=10) #--> publish X: rad_X, Y: rad_Y, Z:height

        #load model.pth from checkpoint
        self.model = CNNRegressor2D(cfg)
        self.model.load_state_dict(torch.load(os.path.join(cfg.model_directory, 'model.pth')))
        self.model.eval()

        self.debug_trial_counter = 0

    def audio_callback(self, msg, topic):
        """
        Callback method for all the audio topics. Topic name is used to lookup dictionary for the rolling buffer.

        Args:
            msg: The incoming ROS message containing audio data.
        """
        # print(f"topic: {topic} and msg size: {len(msg.data)}")
        
        # Lock the CB function to prevent race conditions that cause N topics to mess up rolling buffer updates
        with self.lock:

            # Get the rolling buffer for the current topic
            rolling_buffer = self.rolling_buffers[topic]

            #buffer the incoming data to the desired length
            input_data = np.asarray(msg.data).reshape((-1,self.num_channels))

            ## Append the input_data to the rolling buffer
            rolling_buffer = np.concatenate((rolling_buffer, input_data))


            # If the buffer is longer than the desired length 
            if rolling_buffer.shape[0] > int(self.fs * self.buffer_duration):
                #trim the oldest part of the buffer by the size of the newly appended input
                rolling_buffer = rolling_buffer[-int(self.fs * self.buffer_duration):]
            
            # Update the rolling buffer
            self.rolling_buffers[topic] = rolling_buffer
            

    def load_xy_single_trial(self, cfg, trial_n):
        """
        Load the data from a single trial
        """

        num_mics = len(self.cfg.device_list)

        wavs = []
        melspecs = []

        for i in range(num_mics):
            wav_filename = f"{self.dir[trial_n]}/mic{self.cfg.device_list[i]}.wav"

            if i == 0:
                print(f"loading wav file: {wav_filename}")

            wav, sample_rate = torchaudio.load(wav_filename)
            self.sample_rate = sample_rate
            # print(f"sample rate: {sample_rate}") #--> sample rate: 44100

            #to ensure same wav length, either pad or clip to be same length as cfg.max_num_frames
            wav = mic_utils.trim_or_pad(wav, self.cfg.max_num_frames)


            #append to list of wavs
            wavs.append(wav.squeeze(0)) # remove the dimension of size 1

            #apply transform to wav file
            if self.audio_transform:
                mel = self.audio_transform(self.cfg, wavs[i].float())
                melspecs.append(mel.squeeze(0)) # remove the dimension of size 1

        # stack wav files into a tensor of shape (num_mics, num_samples)
        wav_tensor = torch.stack(wavs, dim=0)
        # print(f"dimension of wav tensor: {wav.size()}") #--> dimension of wav tensor: torch.Size([6, 88200])

        #stack mel spectrograms into a tensor of shape (num_mics, num_mels, num_samples)
        mel_tensor = torch.stack(melspecs, dim=0)
        # print(f"size of mel_tensor: {mel_tensor.size()}") #--> size of data: torch.Size([6, 16, 690])

        if self.cfg.audio_transform == 'mel':
            data = mel_tensor

        #get label from directory 
        label_file = f"{self.dir[trial_n]}/gt_label.npy"
        label = np.load(label_file) #--> [distance along cylinder, joint 6] i.e. [0.0 m, -2.7 radian]

        #convert label m unit to cm
        label[0] = label[0] * 100

        x,y  = np.cos(label[1]), np.sin(label[1])
        label[1] = x
        label = np.append(label, y) #--> [height, x, y]

        return data, label

    def process_data(self):
        """
        Process the audio data in the rolling buffer.
        0. Pack the audio data from rolling buffer
        1. Check for collision event
        2. Transform wav data to mel spectrogram
        return the mel spectrogram data
        """


        #put rolling buffer into a list
        data_list = [self.rolling_buffers[topic] for topic in self.rolling_buffers]

        

        # create a smaller window for quickly checking for collision
        sub_window_size = int(self.fs * self.sample_duration)//self.main_loop_rate
        # Calculate the start and end indices for the sub-window
        start_index = (self.rolling_buffers['/audio0'].shape[0] - sub_window_size) // 2
        end_index = start_index + sub_window_size

        # Create the sub-window from the rolling buffer
        sub_window_data = self.rolling_buffers['/audio0'][start_index:end_index]

        #check collision by enveloping the audio data
        has_collision = self.check_for_contact_event(sub_window_data)

        #transform wav data to mel spectrogram only upon collision
        if has_collision:

            # center around the max spike in the audio data 
            trimmed_audio_list = mic_utils.trim_audio_around_peak(data_list, self.fs, self.sample_duration)

            # mic_utils.plot_time_domain(trimmed_audio_list, self.fs)

            #convert audio list to tensor
            tensor_list = [torch.from_numpy(np_array) for np_array in trimmed_audio_list]

            wavs = []
            melspecs = []

            for i in range(self.num_mics):
                wav = mic_utils.trim_or_pad_single(tensor_list[i], self.cfg.max_num_frames)

                #append to list of wavs
                wavs.append(wav.squeeze(1)) # remove the dimension of size 2 to 1

                if self.cfg.subtract_background:
                    pass

                #convert wav data to mel spectrogram
                if self.audio_transform:
                    mel = self.audio_transform(self.cfg, wavs[i].float())
                    melspecs.append(mel.squeeze(0)) # remove the dimension of size 1

            wav_tensor = torch.stack(wavs, dim=0)
            # print(f"dimension of wav tensor: {wav_tensor.size()}") #--> dimension of wav tensor: torch.Size([6, 44100])

            #stack mel spectrograms into a tensor of shape (num_mics, num_mels, num_samples)
            mel_tensor = torch.stack(melspecs, dim=0)
            # print(f"size of mel_tensor: {mel_tensor.size()}") #--> size of data: torch.Size([6, 50, 345])

            if self.cfg.audio_transform == 'mel':
                data = mel_tensor

            if self.cfg.audio_transform == 'wav':
                data = wav_tensor

            
            #normalize the raw input data using the mean,var from the training dataset
            meanvar_path = os.path.join(self.cfg.data_dir, 'meanvar.npy')
            meanvar_np = np.load(meanvar_path) #--> dimension [6,1,1,1] and [6,1,1,1] stacked together


            mean, var = meanvar_np[0], meanvar_np[1]
            data = (data - mean) / np.sqrt(var)

            #unsqueeze to add batch dimension [1, num_mics, num_mels, num_samples]
            data = data.unsqueeze(0)

            label = [0, 0, 0]

            return data, label
        
        else:
            return None, None


        

    def check_for_contact_event(self, audio_data):
        """
        Checks for a contact event using time-domain - amplitude envelope of the audio data. 

        Args:
            audio_data: The audio data list of len num_mics, each with shape (num_samples).

        Returns:
            bool: True if a contact event is detected, False otherwise.
        """
        
        frame_size = 128
        hop_length = 64
        num_mic = len(audio_data)
        collision_threshold = self.cfg.collision_threshold #threshold for collision event for mic0, no contact values are 0.0005 

        
        mic_envelop = mic_utils.amplitude_envelope(audio_data, frame_size, hop_length)
        # mic_utils.plot_envelope_from_signal(audio_data[0], frame_size, hop_length) # --> quick visualization of envelope

        #check for collision event. Collision is when the envelope is above a certain threshold
        if np.max(mic_envelop) > collision_threshold:
            print(f"**** collision detected **** ")

            #publish contact event
            self.pub_contactevent.publish(1)
            return True
        
        return False  

    def predict_location(self, processed_audio):
        """
        Predict the contact location using the trained model.
        Publish the contact location as a Point message [publish X: rad_X, Y: rad_Y, Z:height]
        """
        mic_utils.plot_spectrogram_with_cfg(self.cfg, processed_audio, self.fs)
        Y_pred = self.model(processed_audio) 
        print(f"Y_pred: {Y_pred}")

        #publish contact location
        contact_pt = Point()
        contact_pt.x = Y_pred[0][1].item()
        contact_pt.y = Y_pred[0][2].item()
        contact_pt.z = Y_pred[0][0].item()
        self.pub_contactloc.publish(contact_pt)

    def predict_location_eval(self, processed_audio, label):
        """
        Predict the contact location using the trained model.
        Publish the contact location as a Point message [publish X: rad_X, Y: rad_Y, Z:height]
        """
        #print shape
        print(f" processed_audio shape: {processed_audio.shape}")
        #squeeze batch dimension
        processed_audio_squeezed = processed_audio.squeeze(0)
        # mic_utils.plot_spectrogram_with_cfg(self.cfg, processed_audio_squeezed, self.fs)
        Y_pred = self.model(processed_audio) 
        print(f"Y_pred: {Y_pred}")

        #publish contact location
        contact_pt = Point()
        contact_pt.x = Y_pred[0][1].item()
        contact_pt.y = Y_pred[0][2].item()
        contact_pt.z = Y_pred[0][0].item()
        self.pub_contactloc.publish(contact_pt)

        # print(f"ground truth: {label}")

        #height error
        height_error = np.abs(Y_pred[0][0].item() - label[0])
        # print(f"height error: {height_error}")


@hydra.main(version_base='1.3',config_path='../learning/configs', config_name = 'inference')
def main(cfg: DictConfig):
    print(f" ------ starting audio monitor ------  ")
    rospy.init_node('audio_monitor')
    audio_topics = ['/audio0', '/audio1', '/audio2', '/audio3', '/audio4', '/audio5']  #add additional audio topics if needed
    # audio_topics = ['/audio0', '/audio1', '/audio2']  #add additional audio topics if needed


    fs = 44100  # Sample rate of the audio data
    buffer_duration = 2.0  # duration of the rolling buffer to keep last previous samples (in sec)
    sample_duration = 1.0  # duration of the audio window for processing (in sec)
    
    #rate of main loop
    main_loop_rate = 10
    rate = rospy.Rate(main_loop_rate)

    audio_monitor = AudioMonitor(cfg, audio_topics, fs, buffer_duration, sample_duration, main_loop_rate)

    #sleep a few seconds for the audio buffers to fill up
    rospy.sleep(3)
    
    while not rospy.is_shutdown():

        
        # print(f"size of rolling buffer: {audio_monitor.rolling_buffers['/audio0'].shape}")

        #retreive processed data from buffer
        processed_audio, label = audio_monitor.process_data()

        if processed_audio is not None:
            # visualize the data before feeding into the model
            # processed_audio_squeezed = processed_audio.squeeze(0)
            # mic_utils.plot_spectrogram_with_cfg(cfg, processed_audio_squeezed, fs) 

            #predict the contact location
            audio_monitor.predict_location_eval(processed_audio, label)

        


        rate.sleep()

    rospy.spin()

if __name__ == '__main__':
    main()
    