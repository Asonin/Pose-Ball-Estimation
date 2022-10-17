import os
import time
from moviepy.editor import VideoFileClip

special = ['1009_11','1009_12','1009_13','1009_14','1009_15','1014_10','1014_11',
           '1014_12','1014_13','1014_14','1014_15','1014_16','1014_17','1014_18',
           '1014_19','1014_20','1014_21']

def video_duration_2(filename):
    clip = VideoFileClip(filename)
    return float(clip.duration)

# sequence = input()
dir_base = f'/home1/zhuwentao/projects/multi-camera/mvball_proj/output/wusi/'
file = os.listdir(dir_base)
total_length = 0
sequence_calc = {}
for folder in file: # e.g. 1002_2
    p = dir_base + folder + '/'
    sequences = os.listdir(p)
    sequence_length = 0
    for sequence in sequences: # e.g. 1
        p1 = p + sequence + '/'
        p2 = p1 + '3d.mp4'
        
        # t = video_duration_2(p2)
        # sequence_length += t
        # total_length += t
        if not os.path.exists(p2):
            print(p2)
    # if folder in special:
    #     ratio = sequence_length / 120
    # else:
    #     ratio = sequence_length / 180
    # sequence_calc[folder] = [sequence_length, ratio]
    
    
    