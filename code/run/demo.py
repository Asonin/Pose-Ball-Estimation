from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import argparse
import os
import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import PIL

import json_tricks as json
from tqdm import tqdm
import copy
import pickle

import _init_paths
from core.config import config
from core.config import update_config
# from core.function import batch_evaluate, calc_ap, calc_mpjpe
from utils.utils import create_logger, load_backbone_panoptic
from utils.cameras_cpu import project_pose
from utils.transforms import get_affine_transform as get_transform

# sys.path.append('../FairMOT/src/')
# sys.path.append('../FairMOT/src/lib/')
# import _init_paths_mot
from tracker.multitracker import JDETracker
from utils.transforms import *
# from utils.transforms import affine_transform, get_scale
import dataset
import models


import ipdb




def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


# panoptic
LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10],
         [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14]]

# coco17
LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
        [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]

# shelf / campus
LIMBS14 = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3], [6, 7], [7, 8], [9, 10],
          [10, 11], [2, 8], [3, 9], [8, 12], [9, 12], [12, 13]]


def get_color(idx):
    # idx += 1
    idx = 1
    idx = idx * 3
    color = ((37 * idx) % 255 / 255.0, (17 * idx) % 255 / 255.0, (29 * idx) % 255 / 255.0)
    color = (color[2], color[1], color[0])
    return color


def get_color_cv2(idx):
    # idx += 1
    idx = 1
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color


def save_2d_3d_poses_campus(inputs, preds, metas, keep_ids, tracked=True):
    dir_name = "campus_seq2"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    frame_id = preds[0]
    pred_joints = preds[1]
    pred_id = preds[2]

    a = 0
    nviews = len(inputs)
    center = metas[0]['center'][0]
    scale = metas[0]['scale'][0]
    trans = get_transform(center, scale, 0, (inputs[0].shape[3], inputs[0].shape[2]), inv=1)
    width, height = center.cpu().numpy() * 2
    width = int(width)
    height = int(height)

    padding = 70
    padding_y = 20
    img = np.ones((3 * height + 3 * padding_y, 3 * width + 4 * padding, 3)) * 255

    max_person = len(pred_joints)
    fig = plt.figure(0, figsize=((3 * width + 6 * padding) / 100.0, (height * 2) / 100.0))
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0,
                        top=1.0)
    ax = plt.subplot(111, projection='3d')
    # pred = preds[0]
    # colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
    # colors = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0)]
    for n in range(max_person):
        joint = pred_joints[n]
        if tracked and pred_id[n] not in keep_ids:
            continue
        # if joint[0, 3] >= 0:
        for k in eval("LIMBS{}".format(len(joint))):
            x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
            y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
            z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
            ax.plot(x, y, z, c=get_color(pred_id[n]), lw=2,
                    marker='o', markerfacecolor='w', markersize=3,
                    markeredgewidth=1.5)
        # ax.text(x=joint[1, 0] + 20, y=joint[1, 1] + 20, z=joint[1, 2] + 20, s=str(pred_id[n]), fontsize=40,
        #         color=get_color(pred_id[n]))

        # ax.text(joint[0, 0],
        #         joint[0, 1],
        #         joint[0, 2] + 100.0, str(joint[0, 4].round(3)), color=colors[int(n % len(colors))])


    # ax.set_xlim(-2200.0, 5800.0)
    # ax.set_ylim(800.0, 8800.0)
    ax.set_xlim(-3000.0, 9000.0)
    ax.set_ylim(-1500.0, 10500.0)
    ax.set_zlim(0.0, 2000.0)
    ax.view_init(elev=40)

    buffer_ = BytesIO()  # using buffer,great way!
    plt.savefig(buffer_, format='png')
    buffer_.seek(0)
    dataPIL = PIL.Image.open(buffer_)
    data = np.asarray(dataPIL)
    buffer_.close()

    # img[(height + padding * 2): (height + padding * 2) + int(2 * height),
    #     padding: padding + width * 3] = data[:, :, [2, 1, 0]]
    img[(height + padding_y * 2): (height + padding_y * 2) + int(2 * height), :] = data[:, padding:-padding, [2, 1, 0]]

    loc = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]
    for k in range(len(inputs)):
        input = inputs[k][0]
        min = float(input.min())
        max = float(input.max())
        input.add_(-min).div_(max - min + 1e-5)
        input = inputs[k][0].flip(0).mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        input = cv2.warpAffine(
            input,
            trans, (width, height),
            flags=cv2.INTER_LINEAR)

        y = loc[k][0] * (height + padding_y) + padding_y
        x = loc[k][1] * (width + padding) + padding
        img[y: y + height, x: x + width] = input

        for n in range(max_person):
            # if preds[0, n, 0, 3] >= 0:
            if tracked and pred_id[n] not in keep_ids:
                continue
            joints = pred_joints[n]
            cam = {}
            for key, v in metas[k]['camera'].items():
                cam[key] = v[0].cpu().numpy()
            pred_2d = project_pose(joints[:, 0:3], cam)
            # gt_2d = project_pose(gts[n, :, 0:3], cam)

            ori_width = int(center[0] * 2)
            ori_height = int(center[1] * 2)
            joints_vis = np.ones(len(pred_2d))
            x_check = np.bitwise_and(pred_2d[:, 0] >= 0,
                                     pred_2d[:, 0] <= ori_width - 1)
            y_check = np.bitwise_and(pred_2d[:, 1] >= 0,
                                     pred_2d[:, 1] <= ori_height - 1)
            check = np.bitwise_and(x_check, y_check)
            joints_vis[np.logical_not(check)] = 0

            pose_2d = pred_2d
            pose_2d[:, 0] = x + pose_2d[:, 0]
            pose_2d[:, 1] = y + pose_2d[:, 1]

            # colors = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0)]
            # colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 0, 0)]
            # colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
            for limb in eval("LIMBS{}".format(len(pose_2d))):
                # color_k = ERROR if k in pred_pair[n]['error'] else RIGHT
                # color_k = colors[n]
                if joints_vis[limb[0]] and joints_vis[limb[1]]:
                    cv2.line(img, tuple(pose_2d[limb[0]].astype(np.int)), tuple(pose_2d[limb[1]].astype(np.int)),
                             color=get_color_cv2(pred_id[n]), thickness=2)

            for joint, joint_vis in zip(pose_2d, joints_vis):
                if joint_vis:
                    cv2.circle(img, (int(joint[0]), int(joint[1])), 2,
                               [255, 255, 255], -1)
            # if joints_vis[1]:
            #     cv2.putText(img, str(pred_id[n]), (int(pose_2d[1, 0] - 30), int(pose_2d[1, 1] - 30)), cv2.FONT_HERSHEY_SIMPLEX,
            #                 3, get_color_cv2(pred_id[n]), 6)

    if tracked:
        file_name = os.path.join(dir_name, f"frame_tracked_{frame_id}.jpg")
    else:
        file_name = os.path.join(dir_name, f"frame_{frame_id}.jpg")
    cv2.imwrite(file_name, img)


def save_2d_3d_poses(inputs, preds, metas, keep_ids, tracked=True):
    dir_name = "output/demo_results"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    frame_id = preds[0]
    pred_joints = preds[1]
    pred_id = preds[2]

    a = 0
    nviews = len(inputs)
    center = metas[0]['center'][0]
    scale = metas[0]['scale'][0]
    trans = get_transform(center, scale, 0, (inputs[0].shape[3], inputs[0].shape[2]), inv=1)
    width, height = center.cpu().numpy() * 2
    width = int(width)
    height = int(height)

    padding = 30
    img = np.ones((3 * height + 4 * padding, 3 * width + 4 * padding, 3)) * 255

    max_person = len(pred_joints)
    fig = plt.figure(0, figsize=(width/50, height/50))
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0,
                        top=1.0)
    ax = plt.subplot(111, projection='3d')
    # pred = preds[0]
    # colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
    # colors = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0)]
    for n in range(max_person):
        joint = pred_joints[n]
        if tracked and pred_id[n] not in keep_ids:
            continue
        # if joint[0, 3] >= 0:
        for k in eval("LIMBS{}".format(len(joint))):
            x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
            y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
            z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
            ax.plot(x, y, z, c=get_color(pred_id[n]), lw=6,
                    marker='o', markerfacecolor='w', markersize=10,
                    markeredgewidth=4)
        # ax.text(x=joint[1, 0] + 20, y=joint[1, 1] + 20, z=joint[1, 2] + 20, s=str(pred_id[n]), fontsize=40,
        #         color=get_color(pred_id[n]))

        # ax.text(joint[0, 0],
        #         joint[0, 1],
        #         joint[0, 2] + 100.0, str(joint[0, 4].round(3)), color=colors[int(n % len(colors))])


    ax.set_xlim(-3000.0, 3000.0)
    ax.set_ylim(-3000.0, 3000.0)
    ax.set_zlim(0.0, 3000.0)
    # ax.view_init(elev=50, azim=60)

    buffer_ = BytesIO()  # using buffer,great way!
    plt.savefig(buffer_, format='png')
    buffer_.seek(0)
    dataPIL = PIL.Image.open(buffer_)
    data = np.asarray(dataPIL)
    buffer_.close()

    img[(height + padding * 2): (height + padding * 2) + 2 * height,
        (width + padding * 2): (width + padding * 2) + width * 2] = data[:, :, [2, 1, 0]]

    loc = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]
    for k in range(len(inputs)):
        input = inputs[k][0]
        min = float(input.min())
        max = float(input.max())
        input.add_(-min).div_(max - min + 1e-5)
        input = inputs[k][0].flip(0).mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        input = cv2.warpAffine(
            input,
            trans, (width, height),
            flags=cv2.INTER_LINEAR)

        x = loc[k][0] * (height + padding) + padding
        y = loc[k][1] * (width + padding) + padding
        img[x: x + height, y: y + width] = input

        for n in range(max_person):
            # if preds[0, n, 0, 3] >= 0:
            if tracked and pred_id[n] not in keep_ids:
                continue
            joints = pred_joints[n]
            cam = {}
            for key, v in metas[k]['camera'].items():
                cam[key] = v[0].cpu().numpy()
            pred_2d = project_pose(joints[:, 0:3], cam)
            # gt_2d = project_pose(gts[n, :, 0:3], cam)

            ori_width = int(center[0] * 2)
            ori_height = int(center[1] * 2)
            joints_vis = np.ones(len(pred_2d))
            x_check = np.bitwise_and(pred_2d[:, 0] >= 0,
                                     pred_2d[:, 0] <= ori_width - 1)
            y_check = np.bitwise_and(pred_2d[:, 1] >= 0,
                                     pred_2d[:, 1] <= ori_height - 1)
            check = np.bitwise_and(x_check, y_check)
            joints_vis[np.logical_not(check)] = 0

            pose_2d = pred_2d
            pose_2d[:, 0] = y + pose_2d[:, 0]
            pose_2d[:, 1] = x + pose_2d[:, 1]

            # colors = [(0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0)]
            # colors = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255), (255, 0, 0)]
            # colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
            for limb in eval("LIMBS{}".format(len(pose_2d))):
                # color_k = ERROR if k in pred_pair[n]['error'] else RIGHT
                # color_k = colors[n]
                if joints_vis[limb[0]] and joints_vis[limb[1]]:
                    cv2.line(img, tuple(pose_2d[limb[0]].astype(np.int)), tuple(pose_2d[limb[1]].astype(np.int)),
                             color=get_color_cv2(pred_id[n]), thickness=6)

            for joint, joint_vis in zip(pose_2d, joints_vis):
                if joint_vis:
                    cv2.circle(img, (int(joint[0]), int(joint[1])), 6,
                               [255, 255, 255], -1)
            # if joints_vis[1]:
            #     cv2.putText(img, str(pred_id[n]), (int(pose_2d[1, 0] - 30), int(pose_2d[1, 1] - 30)), cv2.FONT_HERSHEY_SIMPLEX,
            #                 3, get_color_cv2(pred_id[n]), 6)

    if tracked:
        file_name = os.path.join(dir_name, f"frame_tracked_{frame_id}.jpg")
    else:
        file_name = os.path.join(dir_name, f"frame_{frame_id}.jpg")
    cv2.imwrite(file_name, cv2.resize(img, (0, 0), fx=0.5, fy=0.5))

def main():
    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(
            config, args.cfg, 'demo')
    cfg_name = os.path.basename(args.cfg).split('.')[0]
    gpus = [int(i) for i in config.GPUS.split(',')]

    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, "validation", False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file %s for testing!' % test_model_file)

    model.eval()

    results = []
    frame_id = 0
    tracker = JDETracker(frame_rate=30)
    preds = []
    # preds = pickle.load(open("campus_200_1400.pkl", "rb"))
    with torch.no_grad():
        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(tqdm(test_loader)):
            ipdb.set_trace()
            # if i < 315:
            #     continue
            if 'panoptic' in config.DATASET.TEST_DATASET:
                pred, heatmaps, _, grid_centers, _, _, _ = model(views=inputs, meta=meta)
                pred = pred.detach().cpu().numpy()
            # elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
            else:
                pred, heatmaps, grid_centers, _, _, _ = model(meta=meta, input_heatmaps=input_heatmap)
                pred_coco = pred.detach().cpu().numpy()
                pred = np.zeros((pred.shape[0], pred.shape[1], 17, pred.shape[3]), dtype=np.float32)
                for b in range(pred.shape[0]):
                    for n in range(pred.shape[1]):
                        pred[b, n, :, :3] += eval(f'coco2{config.DATASET.TEST_DATASET.split("_")[0]}3D')(
                            pred_coco[b, n, :, :3])
                        pred[b, n, :, 3:] += pred_coco[b, n, 0, 3:]
            pred = pred[0]
            pred = pred[pred[:, 0, 3] >= 0, :, :]
            preds.append(pred)
			
        for i in tqdm(range(len(preds))):
            pred = preds[i][:, :, :3]
            # pred = np.array([coco2shelf3D(p) for p in preds[i]])
            if len(pred) == 0:
                results.append((frame_id + 1, [], []))
                frame_id += 1
                continue
            # score = preds[i][:, 0, 4]
            # print(i, score)
            # pred = pred[score > 0.2]
            embeddings = np.zeros((pred.shape[0], 64), dtype=np.float32)
            # pred[:, 0, :] = ((pred[:, 1, :] + pred[:, 2, :]) / 2.0 + pred[:, 0, :]) / 2.0
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
            # print('time consuming: ' + str(timer.average_time * 1000.0))

            # save results
            results.append((frame_id + 1, online_joints, online_ids))
            frame_id += 1
        keep_ids = []
        for t in tracker.tracked_stracks + tracker.lost_stracks + tracker.removed_stracks:
            if t.frame_length > 100:
                keep_ids.append(t.track_id)
                print(t.track_id, t.frame_length, t)


        for i, (inputs, targets_2d, weights_2d, targets_3d, meta, input_heatmap) in enumerate(tqdm(test_loader)):
            # if len(results[i][2]) <= 6:
            #     continue
            save_2d_3d_poses(inputs, results[i], meta, keep_ids, True)

        # pickle.dump(preds, open("campus_200_1400.pkl", "wb"))



if __name__ == "__main__":
    main()