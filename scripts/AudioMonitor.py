#!/usr/bin/env python
import sys
import os
import numpy as np

#ros
import rospy
from std_msgs.msg import String  # Replace with actual audio message type
sys.path.append('/home/iam-lab/audio_localization/catkin_ws/src/sounddevice_ros/msg')
from sounddevice_ros.msg import AudioInfo, AudioData

# from your_nn_model_package import YourNNModel  # Import your neural network model class

""""
Class for simply monitoring audio data from ROS topics and then outputting processed data.
This class is intended to be used stand-alone as if it's a sensor node.

Author: Mark Lee (MoonRobotics@cmu.edu)
Created: 2024-04-09
"""
class AudioMonitor:
    def __init__(self, audio_topics, fs, sample_duration, sample_overlap_ratio):
        """
        Initializes the AudioMonitor class with the given audio topics.
        
        Args:
            audio_topics (list of str): List of ROS topic names to subscribe to for audio data.
        """
        self.fs = fs  # Sample rate of the audio data
        self.num_channels = 1 # Number of audio channels
        self.sample_duration = sample_duration # Size of the audio window for processing (in seconds)
        self.num_mics = len(audio_topics) # Number of microphones

        # Ratio of overlap between consecutive samples
        self.sample_overlap_ratio = sample_overlap_ratio #if 0.5, then 50% overlap --> publish freq is 2*sample_duration, if 1.0, then no overlap --> publish freq is same as sample_duration
        self.rolling_buffers = {topic: np.zeros((1, self.num_channels)) for topic in audio_topics}


        self.audio_subscribers = []
        for topic in audio_topics:
            sub = rospy.Subscriber(topic, AudioData, lambda msg, topic=topic: self.audio_callback(msg, topic), queue_size=10)
            self.audio_subscribers.append(sub)

        #publishers
        self.pub = rospy.Publisher('contact_event', String, queue_size=10)


        # Load your neural network model
        # self.nn_model = YourNNModel()  # Adjust this according to how your model is initialized

        # ROS publisher for the contact location
        # self.contact_loc_publisher = rospy.Publisher('/contact_location', String, queue_size=10)

    def audio_callback(self, msg, topic):
        """
        Callback method for processing incoming audio data.

        Args:
            msg: The incoming ROS message containing audio data.
        """
        # print(f"topic: {topic} and msg size: {len(msg.data)}")

        # Get the rolling buffer for the current topic
        rolling_buffer = self.rolling_buffers[topic]

        #buffer the incoming data to the desired length
        input_data = np.asarray(msg.data).reshape((-1,self.num_channels))
        # print(f"self.rolling_buffer shape[0]: {self.rolling_buffer.shape[0]} input_data shape: {input_data.shape}")

        ## Append the input_data to the rolling buffer
        rolling_buffer = np.concatenate((rolling_buffer, input_data))

        # Check if the buffer is full
        if rolling_buffer.shape[0] > int(self.fs * self.sample_duration):
            
            # CHECK only 1 audio topic to see if full, for processing. 
            # DON'T want separate processing from all the individual topics
            print(f"topic {topic} is full")
            if topic == '/audio0':
                # self.process_data()
            
                #publish contact event
                self.pub.publish("contact_event")

            # Slide the buffer by the overlap ratio
            overlap_samples = int(self.fs * self.sample_duration * self.sample_overlap_ratio)
            rolling_buffer = rolling_buffer[overlap_samples:]

        # Update the rolling buffer
        self.rolling_buffers[topic] = rolling_buffer



        # Process the audio data and check for a contact event
        contact_event = self.check_for_contact_event(msg)
        if contact_event:
            # Call inference on your neural network model
            contact_location = self.nn_model.infer_contact_location(msg)
            # Publish the contact location
            self.contact_loc_publisher.publish(contact_location)

    def check_for_contact_event(self, audio_data):
        """
        Checks for a contact event in the given audio data.

        Args:
            audio_data: The audio data to be analyzed.

        Returns:
            bool: True if a contact event is detected, False otherwise.
        """
        # Implement your logic to check for contact event
        # This could be a simple threshold check, a machine learning model, etc.
        # Example:
        # return some_condition(audio_data)
        pass

    def process_data(self):
        """
        Process the audio data in the rolling buffer.
        1. Pack the audio data from rolling buffer
        2. Transform wav data to mel spectrogram
        3. visualize the mel spectrogram
        """
        
        


if __name__ == '__main__':
    print(f" ------ starting audio monitor ------  ")
    rospy.init_node('audio_monitor')
    audio_topics = ['/audio0', '/audio1', '/audio2', '/audio3', '/audio4', '/audio5']  #add additional audio topics if needed

    fs = 44100  # Sample rate of the audio data
    sample_duration = 1.0  # duration of the audio window for processing (in sec)

    #if 0.5, then 50% overlap --> publish freq is 2*sample_duration, 
    #if 1.0, then no overlap --> publish freq is same as sample_duration
    sample_overlap_ratio = 1.0  # ratio of overlap between consecutive samples (0 to 1)

    audio_monitor = AudioMonitor(audio_topics, fs, sample_duration, sample_overlap_ratio)
    rospy.spin()