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
import librosa

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
    '-dlist', '--devicelist', nargs='+', type=int,  default=[9,10,11,12],
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
  
    # get audio data from .wav file
    file_name0 = f"rec_device_{args.devicelist[0]}.wav"
    file_name1 = f"rec_device_{args.devicelist[1]}.wav"
    file_name2 = f"rec_device_{args.devicelist[2]}.wav"
    file_name3 = f"rec_device_{args.devicelist[3]}.wav"
    print(f" file_name0: {file_name0} ")


    fs, data0 = read(file_name0)
    fs, data1 = read(file_name1)
    fs, data2 = read(file_name2)
    fs, data3 = read(file_name3)

    data0, sr = librosa.load(file_name0)
    data1, sr = librosa.load(file_name1)  
    data2, sr = librosa.load(file_name2)
    data3, sr = librosa.load(file_name3)


    # # data0 = wave.open(file_name0, "r")
    # # data1 = wave.open(file_name1, "r")

    print(f"size of data0: {len(data0)}, size of data1: {len(data1)}")

    # trim data0,1,2,3 to the shortest length
    min_length = min(len(data0), len(data1), len(data2), len(data3))
    data0 = data0[:min_length]
    data1 = data1[:min_length]
    data2 = data2[:min_length]
    data3 = data3[:min_length]
    


    # plot data0 and data1 in same plot
    # Create a time array for plotting
    time = np.arange(0, len(data0)) / fs

    plt.figure(figsize=(10, 6))

    plt.plot(time, data0, label='data0')
    plt.plot(time, data1, label='data1')
    plt.plot(time, data2, label='data2')
    plt.plot(time, data3, label='data3')

    plt.title('Audio Data')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.show()
    


    print(f" ------ ending script ------  ")
    

#init main function

if __name__ == '__main__':
    main()