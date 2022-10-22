# -*- coding: utf-8 -*-
import os
#设定文件路径
path= '/home1/zhuwentao/projects/multi-camera/mvball_proj/output/wusi/1021_117_烂了/2/'
dir_3d = path + '3d_vis/'
dir_v = path + 'image_with_projected_ball_and_poses/'#对目录下的文件进行遍历
# for file in os.listdir(dir_v):
# #判断是否是文件
#     if os.path.isfile(os.path.join(dir_v,file))==True:
# #设置新文件名
#         serial = int(file[5:13])
#         print(serial)
#         serial = serial - 26
#         ss = '{}_{:08}'.format('test', serial)
#         view = file[13:]
#         # print(view)
#         name = ss + view
#         # print(name)
#         new_name=file.replace(file,name)
# #重命名
#         os.rename(os.path.join(dir_v,file),os.path.join(dir_v,new_name))
# #结束

#对目录下的文件进行遍历
for i in range(1,12):
    for file in os.listdir(dir_v):
    #判断是否是文件
        if os.path.isfile(os.path.join(dir_v,file))==True:
            if file[-6] == '_':
                s = int(file[-5])
            else:
                s = int(file[-6]+file[-5])
            if s != i:
                continue
            # print(s)
    #设置新文件名
            serial = int(file[5:13])
            # print(serial)
            serial = serial - 26
            ss = '{}_{:08}'.format('test', serial)
            view = file[13:]
            # print(view)
            name = ss + view
            # print(name)
            new_name=file.replace(file,name)
    #重命名
            os.rename(os.path.join(dir_v,file),os.path.join(dir_v,new_name))
#结束
print ("End")