from asyncore import write
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
# total_14 = 0
# total_09 = 0
sequence_calc = {}
# 先创建并打开一个文本文件
f = open('legth_calc.txt', 'w')

for folder in file: # e.g. 1002_2
    # print(int(folder[2]))
    # x = int(folder[2])
    p = dir_base + folder + '/'
    sequences = os.listdir(p)
    sequence_length = 0
    for sequence in sequences: # e.g. 1
        p1 = p + sequence + '/'
        p2 = p1 + '3d.mp4'
        if os.path.exists(p2):
            t = video_duration_2(p2)
            sequence_length += t
            # if x == 1:
            #     total_14 += t
            # else:
            #     total_09 += t
            total_length += t
        else:
            print(p2)
    # if folder in special:
    #     ratio = sequence_length / 120
    # else:
    #     ratio = sequence_length / 180
    sequence_calc[folder] = [sequence_length, 1]
    print(folder,sequence_calc[folder])
    # f.write(str(folder)+' '+str(sequence_calc[folder])+'\n')


print('total_length = ',total_length)
ratio = total_length / float(40*60)
# ratio = total_length / float(17*300)
print('total_ratio = ',ratio)
# ratio_14 = total_14 / (float(17*300) - float(34 * 60))
# print('total_14 = ',total_14)
# print('ratio_14 = ',ratio_14)
# ratio_09 = total_09 / float(34 * 60)
# print('total_09 = ',total_09)
# print('ratio_09 = ',ratio_09)
      
f,write(str(total_length)+' '+str(ratio)+'\n')
# f.write(str(total_14)+' '+str(ratio_14)+'\n')
# f.write(str(total_09)+' '+str(total_09)+'\n')
	
# 注意关闭文件
f.close()
    
    