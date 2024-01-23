#!/usr/bin/env python3
"""Plot the live microphone signal(s) with matplotlib.

Matplotlib and NumPy have to be installed.

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

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')

parser.add_argument(
    '-dlist', '--devicelist', nargs='+', type=int,  default=[9,10,11,12,13,14],
    help='-dlist 9 10 11 12')

parser.add_argument(
    '-w', '--window', type=float, default=200, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')


args = parser.parse_args(remaining)
if any(c < 1 for c in args.channels):
    parser.error('argument CHANNEL: must be >= 1')
mapping = [c - 1 for c in args.channels]  # Channel numbers start with 1



def audio_callback(q):

    def callback(data, frames, time, status):
        if status:
            print(status)
        q.put(data.copy())

    return callback


def main():
    print(f" ------ starting script ------  ")
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        args.samplerate = device_info['default_samplerate']

    #get number pf devicelist
    mic_number = len(args.devicelist)



    queue_list = []
    stream_sd_list = []
    SoundFile_list = []

    
    for i in range(mic_number):
        #create instance of queue for each device
        queue_name = f"q{i}" #q0 = queue.Queue() ... q6 = queue.Queue()
        queue_name = queue.Queue() 
        queue_list.append(queue_name)

        #create instance of stream for each device
        stream_file = sd.InputStream(
            device=args.devicelist[i], channels=max(args.channels),
            samplerate=args.samplerate, callback=audio_callback(queue_list[i]))

        stream_sd_list.append(stream_file)


        #if file exists, delete it and recreate it
        try:
            os.remove(f"rec_device_{args.devicelist[i]}.wav")
        except OSError:
            pass

        #create soundfile for each device
        sf = SoundFile(
            file=f"rec_device_{args.devicelist[i]}.wav",
            mode="x",
            samplerate=int(stream_sd_list[i].samplerate),
            channels=stream_sd_list[i].channels,
        )

        SoundFile_list.append(sf)

    
    # record audio for all 6 streams until KeyboardInterrupt
    with contextlib.ExitStack() as stack:
        for stream in stream_sd_list:
            stack.enter_context(stream)
        print("press Ctrl+C to stop the recording")
        try:
            while True:
                for i in range(6):
                    SoundFile_list[i].write(queue_list[i].get())
        except KeyboardInterrupt:
            print("\nInterrupted by user.")


    # record audio for all 6 streams until KeyboardInterrupt
    # with stream0, stream1, stream2, stream3:
    #     print("press Ctrl+C to stop the recording")
    #     try:
    #         while True:
    #             sf0.write(q0.get())
    #             sf1.write(q1.get())
    #             sf2.write(q2.get())
    #             sf3.write(q3.get())

    #     except KeyboardInterrupt:
    #         print("\nInterrupted by user.")
    

    
    #sample rate
    print(f" stream1.samplerate: {stream_sd_list[0].samplerate} ")

    #size of sf0
    print(f" size of sf0: {SoundFile_list[0].tell()} ")



    print(f" ------ ending script ------  ")
    

#init main function

if __name__ == '__main__':
    main()