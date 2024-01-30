# audio_datacollection

## setup
pip install librosa
pip install sounddevice
pip install pysoundfile

## initial setup
See what mic devices are available. Running code directory `/home/iam-lab/audio_localization/audio_datacollection/scripts`

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

Generate audio .wav files until keyboard interrupt
```
python3 record_all_mics.py -dlist 9 10 11 12 13 14
```

Visualize plot of .wav files
```
python3 plot_all_mics.py
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
```
python3 record_with_franka.py 
```