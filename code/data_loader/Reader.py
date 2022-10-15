import cv2
import torch

class Reader():
    def __init__(self, camera_ids, matches, aligned_len, image_size, resize_transform, transform, args):
        self.resize_transform = resize_transform
        self.transform = transform
        self.image_size = image_size
        self.frame_num = {}
        self.aligned_len = aligned_len
        self.videocap = {}
        self.camera_ids = camera_ids
        self.matches = matches
        self.prev = {} # save at most five pics
        self.prev_id = {}
        self.flag = False
        for cam_id in camera_ids:
            self.prev_id[cam_id] = []
            self.frame_num[cam_id] = 0
            self.prev[cam_id] = {}
            self.videocap[cam_id] = cv2.VideoCapture('../out/%s/%s.mp4' % (args.sequence, cam_id))
    
    def __call__(self, fid):
        img_list = []
        pose_img_list = []
        
        for k, cam_id in enumerate(self.camera_ids):
            ori_fid = self.matches[fid, k]
            # if cam_id == 10:
            #     print(f"ori_fid = {ori_fid}")
            while(1):
            # while(self.frame_num[cam_id] != ori_fid):
                # if cam_id == 10:
                #     print(f"now the id is {self.frame_num[cam_id]}")
                if not self.videocap[cam_id].isOpened():
                    print(f"cideo_cap {cam_id} closed")
                    break
                ret, frame = self.videocap[cam_id].read()
                if not ret:
                    print("at the end of the video_file")
                    self.flag = True
                    break
                
                self.frame_num[cam_id] += 1
                cur_id = self.frame_num[cam_id]
                 
                if cur_id > ori_fid: 
                    # print(f"using past frame of cam{cam_id} at {cur_id}")
                    # print(f"ori_fid = {ori_fid}")
                    # print(cur_id)
                    img_list.append(self.prev[cam_id][ori_fid])
                    pose_img = cv2.cvtColor(self.prev[cam_id][ori_fid], cv2.COLOR_BGR2RGB)
                    
                elif cur_id == ori_fid:
                    img_list.append(frame)
                    pose_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                self.prev[cam_id][cur_id] = frame
                self.prev_id[cam_id].append(cur_id)
                # print(self.prev_id[cam_id])
                if len(self.prev_id[cam_id]) == 30:
                    self.prev[cam_id].pop(self.prev_id[cam_id][0])
                    del self.prev_id[cam_id][0]
                
                if cur_id < ori_fid: # not found yet
                    continue
                    
                # require extra transformation for poee estimation
                pose_img = cv2.warpAffine(pose_img, self.resize_transform, (int(self.image_size[0]), 
                                        int(self.image_size[1])), flags=cv2.INTER_LINEAR)
                pose_img = self.transform(pose_img)
                pose_img_list.append(pose_img)
                break
                         
        pose_img_list = torch.stack(pose_img_list, dim=0).unsqueeze(0)
        
        return img_list, pose_img_list, self.flag 
    
    def __del__(self):
        for key, value in self.videocap.items():
            print(f"videocap{key} released")
            value.release()
            
   