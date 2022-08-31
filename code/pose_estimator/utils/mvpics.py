import os
import shutil

num_cams = 8
cam_name = ["1", "2", "3", "4", "5", "6", "7", "8"]

for i in range(num_cams):
    src = "/home1/zhuwentao/projects/multi-camera/CH-HCNetSDKV6.1.6.4_build20201231_linux64/consoleDemo/linux64/proj/out/0715_easy/synced/%s" % cam_name[i]
    dest = "data/hikon/0715_easy/%d" % i
    res = shutil.move(src, dest)
    print(res)
