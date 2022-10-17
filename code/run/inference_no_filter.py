from bdb import Breakpoint
from datetime import datetime
from genericpath import exists
from tqdm import tqdm
from cmath import sqrt
from os import path

import numpy as np
import os
import shutil
import torch
import cv2
import pandas as pd
import os
import argparse
import yaml
import argparse
import numpy as np
import math
import _init_paths
import ffmpeg

# modules for ball detection
from ball_detector.models.experimental import attempt_load
from ball_detector.utils.torch_utils import select_device, time_synchronized
from ball_detector.basketball import estimate_ball_3d

# modules for ball detection
from pose_estimator.core.config import config, update_config
from pose_estimator.utils.utils import load_pose_model, get_transform, estimate_pose_3d, get_cam
from pose_estimator.utils.vis import save_3d_images, save_image_with_projected_ball_and_poses

# modules for tracker
from tracker.multitracker import JDETracker

# modules for one-euro filter
from one_euro_filter.filter import OneEuroFilter, slide_window, OneEuroFilterPose, do_interpolation
 
#modules for data_loader
from data_loader.Reader import Reader       


def within(idx, n_frames):
    for i in range(len(idx)):
        if idx[i] >= n_frames[i]:
            return False
    return True

    
def match_timestamps(camera_ids):
    
    num_cams = len(camera_ids)
    ts = []
    dts = []
    n_frames = []

    for i, cam_id in enumerate(camera_ids):
        df = pd.read_csv('../out/%s/%s.csv' % (args.sequence, cam_id), sep=',', header=None)
        cam_log = df.values
        t = [] # timestamp
        dt = []
        for idx, row in enumerate(cam_log):
            if idx <= 1:
                continue
            t.append(int(row[2]))
            #specific time
            y, mo, d, h, mi, s, ms = int(row[3]), int(row[4]), int(row[5]), int(row[6]), \
                                     int(row[7]), int(row[8]), int(row[9])
            # print(y,mo,d,h,mi,s,ms)
            dt.append(datetime(y, mo, d, h, mi, s, ms * 1000).timestamp())  # milisecond to microsecond
        frames = len(dt)
        dts.append(np.array(dt))
        
        ts.append(np.array(t))
        print("Cam %s Done, %d frames." % (cam_id, frames))
        n_frames.append(frames)


    idx = []
    start = [dts[x][51] for x in range(num_cams)]
    lff = max(start)  # Get the latest first frame.
    ref = np.argmax(start) # get the index for the greatest
    # get starting point
    for i in range(num_cams):
        idx.append(np.argmin(abs(dts[i] - lff)))
    idx = np.array(idx) # get starting_index for all the cameras
    matches = []
    
    while within(idx, n_frames): # not crossing the bounds
        for i in range(num_cams):
            # print(i)
            # print(ref)
            while within(idx + 1, n_frames) and (
                    abs(dts[i][idx[i] + 1] - dts[ref][idx[ref]]) < abs(dts[i][idx[i]] - dts[ref][idx[ref]])):
                idx[i] += 1
            while within(idx - 1, n_frames) and (
                    abs(dts[i][idx[i] - 1] - dts[ref][idx[ref]]) < abs(dts[i][idx[i]] - dts[ref][idx[ref]])):
                idx[i] -= 1
        matches.append(idx.copy())
        idx += 1

    matches = np.array(matches)
    # for i in matches:
    #     print(i)
    # cnt = 0
    # prev = matches[0]
    # for i in matches:
    #     if cnt == 0:
    #         cnt += 1
    #         continue
    #     cnt += 1
    #     for j in range(11):
    #         if prev[j] > i[j]:
    #             print("wtf")
    #             print(prev)
    #             print(i)
    #     prev = i
    #     if cnt > 100:
    #         break
    # print(matches)
    aligned_len = len(matches)
    print("Matched sequence length: %d" % aligned_len)
    
    return matches


def cap_pic(matches, camera_ids, resize_transform, transform):
    # Read time-aligned frames
    num_frames = len(matches)
    image_size = config.NETWORK.IMAGE_SIZE
    reader = Reader(camera_ids, matches, num_frames, image_size, resize_transform, transform, args)
    break_flag = False
    
    if args.breakpoint == -1:
        print("no breakpoints, inferencing the whole sequence")
        pbar = tqdm(total=num_frames)
    else:
        break_flag = True
        print(f"the breakpoint's at {args.breakpoint}")
        pbar = tqdm(total=args.breakpoint)

    recon_list = {}
    pose_recon_list = {}
    output_dir_pose = f'../output/{args.scene}/{args.sequence}'

    tracker = JDETracker(frame_rate=25)
    flag = True # filter instance flag
    
    
    # enumerate over all aligned frames
    for fid in range(num_frames):
        # for tests, set breakpoints
        if break_flag:
            if fid > args.breakpoint:
                print(f"ending inference at fid = {fid}")
                break
        
        img_list, pose_img_list, eof_flag = reader(fid)
        if eof_flag:
            print(f"at the end of video, ending at {fid}")
            break
        # print(len(img_list))
        recon = estimate_ball_3d(x, img_list, cameras, args, model_ball, device)
        # for tests, probably print out the recon to see its shape and value
        # print(f"recon goes like \n {recon}")
        
        poses = estimate_pose_3d(model_pose, pose_img_list, meta, our_cameras, resize_transform_tensor)
        poses = poses[:,poses[0,:,0,4]>=config.CAPTURE_SPEC.MIN_SCORE,:,:]
        # for tests, probably print out the poses to see its shape and value
        # print("poses for this frame goes like:")
        # print(poses)
        # print("poses shape be like:")
        # print(poses.shape)
        
        # pose tracking
        pred = poses.copy()[0]
        # print(pred[:, 0, 3], pred[:, 0, 4])
        # pred = pred[pred[:, 0, 3] >= 0, :, :]
        pred = pred[:, :, :3]

        embeddings = np.zeros((pred.shape[0], 64), dtype=np.float32)
        det_conf = np.ones((pred.shape[0], pred.shape[1], 1), dtype=pred.dtype)
        dets = np.concatenate((pred, det_conf), axis=2)
        online_targets = tracker.update(embeddings, dets)
        online_joints = []
        online_ids = []
        for t in online_targets:
             coord = t.joints
             tid = t.track_id
             online_joints.append(coord)
             online_ids.append(tid)
        # for tests, probably print out the joints after tracking to see its shape and value
        # print("online joints goes like:")
        # print(online_joints)
        # print("online joints shape goes like:")
        # print(np.array(online_joints).shape)
        # print("online ids goes like:")
        # print(fid, pred.shape[0], online_ids)
        
        # only save the results when there are both ball and full-size poses
        if recon is not None and len(online_ids) == args.num_people:
            recon_list[fid] = recon
            pose_recon_list[fid] = [online_ids, online_joints]
        pbar.update(1)
        prefix = '{}_{:08}'.format(os.path.join(output_dir_pose, 'test'), fid)
        save_3d_images(config, recon, online_joints, prefix, online_ids)
        save_image_with_projected_ball_and_poses(config, pose_img_list, recon, np.expand_dims(np.array(online_joints), 0), meta, prefix, our_cameras, resize_transform, online_ids)
            
            
    reader.__del__()
 
 
def postprocess():
    sequence = args.sequence

    dir_base = f'/home1/zhuwentao/projects/multi-camera/mvball_proj/output/wusi/{sequence}'
    p1 = dir_base + f'/3d_vis'
    p2 = dir_base + f'/image_with_projected_ball_and_poses'
    
        
    pics_3d = p1 + '/test_%8d_3d.jpg'
    out_dir_3d = dir_base + f'/3d.mp4'
    (
            ffmpeg
            .input(pics_3d, framerate=25)
            .output(out_dir_3d)
            .run()
    )
    for i in range(1,12):
            pics_projected = p2 + f'/test_%8d_view_{i}.jpg'
            out_dir_projected = dir_base + f'/view{i}.mp4'
            (
                ffmpeg
                .input(pics_projected, framerate=25)
                .output(out_dir_projected)
                .run()
            )
        
        
            
    
if __name__ == '__main__':

    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='models/ball_model_best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--sequence', type=str, default='20220427_ball_01')
    parser.add_argument('--breakpoint', type=int, default='-1')
    parser.add_argument('--num_people', type=int, default='5')
    parser.add_argument('--scene', type=str, default='wusi')
    parser.add_argument('--extrinsics_path', type=str,
                        default='../../dataset/extrinsics')
    parser.add_argument('--device', default='8', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    
    args = parser.parse_args()

    # read configuration file
    config_file = "configs/" + args.scene + ".yaml"
    update_config(config_file)

    camera_ids = config.DATASET.CAMERA_ID

    # load camera parameters of lyk
    cameras = {}
    for cam_id in camera_ids:
        x = np.load(os.path.join(args.extrinsics_path, "{}.npz".format(cam_id)))
        info = {}
        mtx, dist, R, T = x['arr_0'], x['arr_1'][0], x['arr_2'], x['arr_3']
        fx, fy, cx, cy = mtx[0][0], mtx[1][1], mtx[0][2], mtx[1][2]
        k = [[dist[0]], [dist[1]], [dist[4]]]
        p = [[dist[2]], [dist[3]]]
        info['m'] = mtx
        info['d'] = dist
        info['r'] = R
        info['t'] = T
        info['k'], info['p'] = k, p
        info['R'], info['T'] = R.tolist(), T.tolist()
        info['fx'], info['fy'], info['cx'], info['cy'] = fx, fy, cx, cy
        cameras[str(cam_id)] = info

    # load model
    device = select_device(args.device)
    print(device)
    half = device != 'cpu'
    model_ball = attempt_load(args.weights, map_location=device)
    if half:
        model_ball.half()
    print("=> Successfully load the model for 3D ball detection...")

    model_file = 'models/pose_model_best.pth.tar'
    model_pose = load_pose_model(config, model_file)

    cam_file = '../data/calibration_wusi.json'
    our_cameras = get_cam(cam_file, args.sequence)
    meta = {'seq': [args.sequence]}
    resize_transform, transform = get_transform(config)
    resize_transform_tensor = torch.tensor(resize_transform, dtype=torch.float, device=device)

    # read timestamps and
    matches = match_timestamps(camera_ids)
    # run pose and ball estimate
    cap_pic(matches, camera_ids, resize_transform, transform)
    
    postprocess()