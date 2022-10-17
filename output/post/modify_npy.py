import numpy as np

sequence = input('sequence:')
serial = input('serial:')
start = int(input('start:'))
end = int(input('end:'))
base_dir = f'/home1/zhuwentao/projects/multi-camera/mvball_proj/output/wusi/{sequence}/{serial}/'
ball_pos_dir = base_dir + 'ball_pos.npy'
pose_id_dir = base_dir + 'pose_id.npy'
poses_dir = base_dir + 'poses.npy'

b = np.load(ball_pos_dir)
p_id = np.load(pose_id_dir)
p = np.load(poses_dir)

b = b[start:end,:]
p_id = p_id[start:end,:]
p = p.reshape(-1,5,15,3)
p = p[start:end,:,:,:]
print(b.shape)
print(p_id.shape)
print(p.shape)
p = p.reshape(1,-1)

np.save(pose_id_dir,p_id)
np.save(poses_dir,p)
np.save(ball_pos_dir, b)