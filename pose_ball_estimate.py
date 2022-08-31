import numpy as np
import cv2
from datetime import datetime
import pandas as pd
import random
import os
import argparse
import yaml
import ipdb
from ball_detector.basketball import estimate_ball_3d

from estimate_pose import load_model, get_cam, estimate_pose_3d, preprocess
from core.config import config
from core.config import update_config
from utils.vis import save_debug_2d_images, save_multi_image_with_projected_poses


def within(idx, n_frames):
    for i in range(len(idx)):
        if idx[i]>=n_frames[i]:
            return False
    return True


def read_and_match_timestamps():
    ts = []
    dts = []
    n_frames = []
    for i, cam in enumerate(name_list):
        df = pd.read_csv('out/%s/%s.csv' % (args.prefix, cam), sep=',', header=None)
        cam_log = df.values
        t = []
        dt = []
        for idx, row in enumerate(cam_log):
            if idx <= 1:
                continue
            t.append(int(row[2]))
            y, mo, d, h, mi, s, ms = int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7]), int(row[8]), int(
                row[9])
            dt.append(datetime(y, mo, d, h, mi, s, ms * 1000).timestamp())  # milisecond to microsecond
        frames = len(dt)
        dts.append(np.array(dt))
        ts.append(np.array(t))
        print("Cam %s Done, %d frames." % (cam, frames))
        n_frames.append(frames)

    fps = 25.0
    interval = 1 / fps

    idx = []
    start = [dts[x][51] for x in range(num_cams)]
    lff = max(start)  # Get the latest first frame.
    ref = np.argmax(start)
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
    # ipdb.set_trace()
    print("Matched sequence length: %d" % len(matches))
    return matches

def cap_pics(matches):

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)  # make new output folder
    for x, match in enumerate(matches):
        img_list = []

        for i, fid in enumerate(match):
            cam = name_list[i]
            # 'out/%s/%s.mp4' % (args.prefix, cam)
            cap = cv2.VideoCapture('out/%s/%s.mp4' % (args.prefix, cam))   # read video
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)    # certain frame
            res, frame = cap.read()

            

            # prepare pics for pose estimation
            img_for_pose = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img_list.append(frame)

            # image_for_pose = cv2.warpAffine(image_for_pose, trans, (int(image_size[0]), int(image_size[1])), flags=cv2.INTER_LINEAR)
            # image_for_pose = transform(image_for_pose)
            # image_for_pose = image_for_pose.unsqueeze(0)

        # weights = "/home1/zhuwentao/projects/multi-camera/CH-HCNetSDKV6.1.6.4_build20201231_linux64/consoleDemo/linux64/proj/ball_detector/models/best.pt"
        # index = weights.find('/')
        # weights = weights[index+1:-2]
        # index = weights.find('/')
        # weights = weights[index + 1:]
        # recon = estimate_ball_3d(img_list, info_all, weights, args)
        # print(recon)

        final_poses, poses, input_heatmap = estimate_pose_3d(config, model, img_list, meta_list)


if __name__ == '__main__':

    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./model/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--prefix', type=str, default='test')
    parser.add_argument('--scene', type=str, default='wusi')
    parser.add_argument('--extrinsics_path', type=str, default='')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args = parser.parse_args()
    update_config(args.cfg)

    print(config)

    gpus = [int(i) for i in config.GPUS.split(',')]

    # read camera data for pic read
    with open("configs/%s.yaml" % args.scene) as f:
        cfg_data = yaml.load(f, Loader=yaml.FullLoader)

    addr_list = cfg_data['ips']
    name_list = [str(x) for x in cfg_data['names']]
    num_cams = len(name_list)

    # load the model for pose estimation
    model_file = 'output/wusi/voxelpose_50/wusi_0412/model_best.pth.tar'
    model = load_model(config, model_file)

    # Load the camera parameters(extrinsics).
    extrinsics_path = args.extrinsics_path
    # extrinsics_path = os.path.join(extrinsics_path, "extrinsics")

    # load camera_info
    info_all = {}
    for cid in range(1, num_cams + 1):
        x = np.load(os.path.join(extrinsics_path, f"{cid}.npz"))
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
        info_all[str(cid)] = info





    p_list = []

    # read timestamps and match pics
    matches = read_and_match_timestamps()

    #
    cap_pics(matches)
