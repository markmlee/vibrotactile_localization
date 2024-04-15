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
from std_msgs.msg import String  # Replace with actual output message type
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
    def __init__(self, cfg, audio_topics, fs, sample_duration, sample_overlap_ratio):
        """
        Initializes the AudioMonitor class with the given audio topics.
        
        Args:
            audio_topics (list of str): List of ROS topic names to subscribe to for audio data.
        """
        self.cfg = cfg
        self.fs = fs  # Sample rate of the audio data
        self.num_channels = 1 # Number of audio channels
        self.sample_duration = sample_duration # Size of the audio window for processing (in seconds)
        self.num_mics = len(audio_topics) # Number of microphones
        self.audio_transform = TRANSFORMS[cfg.audio_transform] # Transform to apply to the audio data

        # Ratio of overlap between consecutive samples
        self.sample_overlap_ratio = sample_overlap_ratio #if 0.5, then 50% overlap --> publish freq is 2*sample_duration, if 1.0, then no overlap --> publish freq is same as sample_duration
        self.overlap_samples = int(self.fs * self.sample_duration * self.sample_overlap_ratio)

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
        self.pub = rospy.Publisher('contact_event', String, queue_size=10)


        # ROS publisher for the contact location
        # self.contact_loc_publisher = rospy.Publisher('/contact_location', String, queue_size=10)

        #load model.pth from checkpoint
        self.model = CNNRegressor2D(cfg)
        self.model.load_state_dict(torch.load(os.path.join(cfg.model_directory, 'model.pth')))
        self.model.eval()

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
            # print(f"self.rolling_buffer shape[0]: {self.rolling_buffer.shape[0]} input_data shape: {input_data.shape}")

            ## Append the input_data to the rolling buffer
            rolling_buffer = np.concatenate((rolling_buffer, input_data))

            # Update the rolling buffer
            self.rolling_buffers[topic] = rolling_buffer
            
            # Check if all buffers are full
            # if full, process the data and then call prediction on model
            if all(buffer.shape[0] > int(self.fs * self.sample_duration) for buffer in self.rolling_buffers.values()):
                # If all buffers are full, process the data
                processed_audio = self.process_data()

                if processed_audio is not None:
                    # mic_utils.plot_spectrogram_with_cfg(self.cfg, processed_audio, self.fs)
                    #unsqueeze to add batch dimension [1, num_mics, num_mels, num_samples]
                    processed_audio = processed_audio.unsqueeze(0)
                    Y_pred = self.model(processed_audio) 
                    print(f"Y_pred: {Y_pred}")
                    

                # Update buffer by sliding all buffers by the overlap ratio
                for topic in self.rolling_buffers:
                    self.rolling_buffers[topic] = self.rolling_buffers[topic][self.overlap_samples:]


    def process_data(self):
        """
        Process the audio data in the rolling buffer.
        0. Pack the audio data from rolling buffer
        1. Check for collision event
        2. Transform wav data to mel spectrogram
        return the mel spectrogram data
        """

        #print buffer sizes for debugging
        for idx,topic in enumerate(self.rolling_buffers):
            # print(f"topic: {topic} and shape: {self.rolling_buffers[topic].shape}")
            self.buffer_sizes[idx] = self.rolling_buffers[topic].shape[0]
        # print(f"buffer_sizes: {self.buffer_sizes}")


        #put rolling buffer into a list
        data_list = [self.rolling_buffers[topic] for topic in self.rolling_buffers]

        # mic_utils.plot_time_domain(data_list, self.fs)

        #check collision by enveloping the audio data
        has_collision = self.check_for_contact_event(data_list)

        #transform wav data to mel spectrogram only upon collision
        if has_collision:

            #trim audio to equal length
            trimmed_audio_list = mic_utils.trim_to_same_length(data_list)

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

            return data
        
        else:
            return None


        

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

        
        mic_envelop = mic_utils.amplitude_envelope(audio_data[0], frame_size, hop_length)
        # mic_utils.plot_envelope_from_signal(audio_data[0], frame_size, hop_length) # --> quick visualization of envelope

        #check for collision event. Collision is when the envelope is above a certain threshold
        if np.max(mic_envelop) > collision_threshold:
            print(f"**** collision detected **** ")
            return True
        
        return False  


@hydra.main(version_base='1.3',config_path='../learning/configs', config_name = 'inference')
def main(cfg: DictConfig):
    print(f" ------ starting audio monitor ------  ")
    rospy.init_node('audio_monitor')
    audio_topics = ['/audio0', '/audio1', '/audio2', '/audio3', '/audio4', '/audio5']  #add additional audio topics if needed
    # audio_topics = ['/audio0', '/audio1', '/audio2']  #add additional audio topics if needed


    fs = 44100  # Sample rate of the audio data
    sample_duration = 1.0  # duration of the audio window for processing (in sec)

    #if 0.5, then 50% overlap --> publish freq is 2*sample_duration, 
    #if 1.0, then no overlap --> publish freq is same as sample_duration
    sample_overlap_ratio = 1.0  # ratio of overlap between consecutive samples (0 to 1)

    audio_monitor = AudioMonitor(cfg, audio_topics, fs, sample_duration, sample_overlap_ratio)
    
    rospy.spin()

if __name__ == '__main__':
    main()
    