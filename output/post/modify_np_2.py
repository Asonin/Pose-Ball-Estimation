import numpy as np
import os

sequence = input('sequence:')
serial = input('serial:')
start = int(input('cut_start:'))
end = int(input('cut_end:'))
base_dir = f'/home1/zhuwentao/projects/multi-camera/mvball_proj/output/wusi/{sequence}/{serial}/'
new_dir = f'/home1/zhuwentao/projects/multi-camera/mvball_proj/output/wusi/{sequence}/0/'
if os.path.exists(new_dir):
    print('new dir already exists')
    
    quit()
os.mkdir(new_dir)

ball_pos_dir = base_dir + 'ball_pos.npy'
ball_pos_dir_1 = new_dir + 'ball_pos.npy'
pose_id_dir = base_dir + 'pose_id.npy'
pose_id_dir_1 = new_dir + 'pose_id.npy'
poses_dir = base_dir + 'poses.npy'
poses_dir_1 = new_dir + 'poses.npy'

b = np.load(ball_pos_dir)
p_id = np.load(pose_id_dir)
p = np.load(poses_dir)

# b = b[start:end,:]
b2 = b[:start,:]
b3 = b[end:,:]

# p_id = p_id[start:end,:]
p_id_2 = p_id[:start,:]
p_id_3 = p_id[end:,:]

p = p.reshape(-1,5,15,3)
# p = p[start:end,:,:,:]
p2 = p[:start,:,:,:]
p3 = p[end:,:,:,:]
# print(b.shape)
# print(p_id.shape)
# print(p.shape)
print(b2.shape)
print(b3.shape)
print('---------')
print(p_id_2.shape)
print(p_id_3.shape)
print('----------')
print(p2.shape)
print(p3.shape)
p = p.reshape(1,-1)

np.save(pose_id_dir_1,p_id_2)
np.save(pose_id_dir,p_id_3)
np.save(poses_dir_1,p2)
np.save(poses_dir,p3)
np.save(ball_pos_dir_1, b2)
np.save(ball_pos_dir, b3)