from bdb import Breakpoint
from datetime import datetime
from genericpath import exists
from tqdm import tqdm
from cmath import sqrt

import numpy as np
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
from one_euro_filter.filter import slide_window, OneEuroFilterPose
        
    
def within(idx, n_frames):
    for i in range(len(idx)):
        if idx[i] >= n_frames[i]:
            return False
    return True


def read_and_match_timestamps(camera_ids):
    num_cams = len(camera_ids)
    ts = []
    dts = []
    n_frames = []

    for i, cam_id in enumerate(camera_ids):
        df = pd.read_csv('../data/%s/%s.csv' % (args.sequence, cam_id), sep=',', header=None)
        cam_log = df.values
        t = []
        dt = []
        for idx, row in enumerate(cam_log):
            if idx <= 1:
                continue
            t.append(int(row[2]))
            y, mo, d, h, mi, s, ms = int(row[3]), int(row[4]), int(row[5]), int(row[6]), \
                                     int(row[7]), int(row[8]), int(row[9])
            dt.append(datetime(y, mo, d, h, mi, s, ms * 1000).timestamp())  # milisecond to microsecond
        frames = len(dt)
        dts.append(np.array(dt))
        ts.append(np.array(t))
        print("Cam %s Done, %d frames." % (cam_id, frames))
        n_frames.append(frames)


    idx = []
    start = [dts[x][51] for x in range(num_cams)]
    lff = max(start)  # Get the latest first frame.
    ref = np.argmax(start)
    # get starting point
    for i in range(num_cams):
        idx.append(np.argmin(abs(dts[i] - lff)))
    idx = np.array(idx)

    matches = []
    while within(idx, n_frames):
        for i in range(num_cams):
            while within(idx + 1, n_frames) and (
                    abs(dts[i][idx[i] + 1] - dts[ref][idx[ref]]) < abs(dts[i][idx[i]] - dts[ref][idx[ref]])):
                idx[i] += 1
            while within(idx - 1, n_frames) and (
                    abs(dts[i][idx[i] - 1] - dts[ref][idx[ref]]) < abs(dts[i][idx[i]] - dts[ref][idx[ref]])):
                idx[i] -= 1
        matches.append(idx.copy())
        idx += 1

    matches = np.array(matches)
    aligned_len = len(matches)
    print("Matched sequence length: %d" % aligned_len)


    # Read time-aligned frames
    all_frames = {}
    for cam_id in camera_ids:
        frame_num = 0
        all_frames[cam_id] = {}
        videocap = cv2.VideoCapture('../data/%s/%s.mp4' % (args.sequence, cam_id))
        while (videocap.isOpened()):
            ret, frame = videocap.read()
            if not ret:
                break
            frame_num += 1
            all_frames[cam_id][frame_num] = frame
        
        videocap.release()
        print("Cam %s Done." % (cam_id))

    return matches, all_frames


def cap_pics(matches, all_frames, camera_ids, resize_transform, transform):
    num_frames = len(matches)
    image_size = config.NETWORK.IMAGE_SIZE
    pbar = tqdm(total=num_frames)

    recon_list = {}
    all_poses = {}
    output_dir_pose = f'../output/{args.scene}/voxelpose_50/{args.sequence}_final'

    tracker = JDETracker(frame_rate=25)
    flag = True # filter instance flag
    break_flag = False
    if args.breakpoint == -1:
        print("no breakpoints, inferencing the whole set")
    else:
        break_flag = True
        print(f"the breakpoint's at {args.breakpoint}")
        
    # enumerate over all aligned frames
    for fid in range(num_frames):
        # for tests, set breakpoints
        if break_flag:
            if fid > break_flag:
                print(f"ending inference at fid = {fid}")
                break
        
        img_list = []
        pose_img_list = []

        for k, cam_id in enumerate(camera_ids):
            ori_fid = matches[fid, k]
            img = all_frames[cam_id][ori_fid]
            img_list.append(img)

            # require extra transformation for poee estimation
            pose_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pose_img = cv2.warpAffine(pose_img, resize_transform, (int(image_size[0]), 
                                      int(image_size[1])), flags=cv2.INTER_LINEAR)
            pose_img = transform(pose_img)
            pose_img_list.append(pose_img)
        pose_img_list = torch.stack(pose_img_list, dim=0).unsqueeze(0)

        recon = estimate_ball_3d(x, img_list, cameras, args, model_ball, device)
        
        # for tests, probably print out the recon to see its shape and value
        # print(f"recon goes like \n {recon}")
        if recon is not None:
            recon_list[fid] = recon
        
        poses = estimate_pose_3d(model_pose, pose_img_list, meta, our_cameras, resize_transform_tensor)
        poses = poses[:,poses[0,:,0,4]>=config.CAPTURE_SPEC.MIN_SCORE,:,:]

        
        # for tests, probably print out the poses to see its shape and value
        # print("poses for this frame goes like:")
        # print(poses)
        # print("poses shape be like:")
        # print(poses.shape)
        all_poses[fid] = poses
        pbar.update(1)

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
        print("online ids goes like:")
        print(fid, pred.shape[0], online_ids)

        if flag:
            #Fc and beta for pose filter
            min_cutoff = 1
            beta = 0
            flag = False
            one_euro_filter_pose = OneEuroFilterPose(fid, np.array(online_joints), online_ids, min_cutoff=min_cutoff,beta=beta)
        else:
            online_joints = one_euro_filter_pose(fid, np.array(online_joints), online_ids)
        # visualization
        prefix = '{}_{:08}'.format(os.path.join(output_dir_pose, 'test'), fid)
        save_3d_images(config, recon, online_joints, prefix, online_ids)
        save_image_with_projected_ball_and_poses(config, pose_img_list, recon, np.expand_dims(np.array(online_joints), 0), meta, prefix, our_cameras, resize_transform, online_ids)
    

    visualize = str(1)  # use which camera to visualize
    # Recompute the parameters.
    rr, jac0 = cv2.Rodrigues(cameras[visualize]["r"])
    tvec = -np.matmul(cameras[visualize]["r"], cameras[visualize]["t"])
    # Project 3D points to the original video.
    video_path = f"../data/{args.sequence}/1.mp4"
    cap = cv2.VideoCapture(video_path)
    out_path = f'./output/wusi/ball'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_path += f"/{args.sequence}.mp4"

    out = cv2.VideoWriter(out_path, cv2.VideoWriter.fourcc('m', 'p', '4', 'v'), 24, (2560, 1440))
    frameid = 0
    ball_pos = []
    # we insert a window filter to clean the outlier
    print("into slide_window for ball_detection")
    res = slide_window(recon_list)
    # res_3d = np.array(res)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        content = frame  # cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        # print(f"frameid of all is {frameid}")
        # if frameid in res:
        if frameid in res:

            # print(f"frameid of draw is {frameid}\n")

            imgpts, jac = cv2.projectPoints(res[frameid], rr, tvec, cameras[visualize]["m"], cameras[visualize]["d"])

            # print(f"imgpts is {imgpts}")
            center = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))

            if 0 <= center[0] < content.shape[1] and 0 <= center[1] < content.shape[0]:
                cv2.circle(content, center, 20, (0, 0, 255), -1)
            ball_pos.append([frameid + 1, int(res[frameid][0]), int(res[frameid][1]), int(res[frameid][2])])

        out.write(content)
        # cv.imshow('frame', content)
        frameid += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()

    # Save the results.
    # final_res: array with shape N*4; each frame with its [frame_id, x, y, z].
    final_res = np.array(ball_pos)

    np.save(f"ball_{args.sequence}.npy", final_res)


if __name__ == '__main__':

    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='models/ball_model_best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--sequence', type=str, default='20220427_ball_01')
    parser.add_argument('--breakpoint', type=int, default='-1')
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

    # read timestamps and match pics
    matches, all_frames = read_and_match_timestamps(camera_ids)

    # load model
    device = select_device(args.device)
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

    # run pose and ball estimate
    cap_pics(matches, all_frames, camera_ids, resize_transform, transform)
