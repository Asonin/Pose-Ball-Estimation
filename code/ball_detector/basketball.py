import argparse
import glob
import logging
import os
import time
from pathlib import Path
import cv2
import math
import torch
import torch.nn as nn
import numpy as np
from numpy import random

from ball_detector.models.experimental import attempt_load
from ball_detector.utils.datasets import LoadStreams, LoadImages
from ball_detector.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from ball_detector.utils.torch_utils import select_device, time_synchronized
from deep_sort import build_tracker
from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem
import scipy.spatial as spt


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def process(img0, size):
    img = letterbox(img0, new_shape=size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    return img0, img


def unfold_camera_param(camera):
    R = camera['r']
    T = camera['t']
    f = np.array([camera['fx'], camera['fy']])
    c = np.array([camera['cx'], camera['cy']])
    k = camera['k']
    p = camera['p']
    return R, T, f, c, k, p


def build_multi_camera_system(cameras, no_distortion=False):
    """
    Build a multi-camera system with pymvg package for triangulation

    Args:
        cameras: list of camera parameters
    Returns:
        cams_system: a multi-cameras system
    """
    pymvg_cameras = []
    for (name, camera) in cameras.items():
        R, T, f, c, k, p = unfold_camera_param(camera)
        camera_matrix = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]], dtype=float)
        proj_matrix = np.zeros((3, 4))
        proj_matrix[:3, :3] = camera_matrix
        distortion = np.array([k[0], k[1], p[0], p[1], k[2]])
        distortion.shape = (5,)
        T = -np.matmul(R, T)
        M = camera_matrix.dot(np.concatenate((R, T), axis=1))
        camera = CameraModel.load_camera_from_M(
            M, name=name, distortion_coefficients=None if no_distortion else distortion)
        if not no_distortion:
            camera.distortion = distortion
        pymvg_cameras.append(camera)
    return MultiCameraSystem(pymvg_cameras)


def triangulate_one_point_ransac(camera_system, points_2d_set):
    """
    Triangulate 3d point in world coordinates with multi-views 2d points.
    And use ransac to optimize.

    Args:
        camera_system: pymvg camera system
        points_2d_set: list of structure (camera_name, point2d)
    Returns:
        points_3d: 3x1 point in world coordinates
    """
    # change the threshold here(mm)
    threshold = 1000
    # get the list of reconstructed points
    points_candidate = []
    all_2D = []
    global camera_num
    camera_num = 11

    for i in range(1, camera_num):
        for j in range(i + 1, camera_num + 1):
            if i in points_2d_set and j in points_2d_set:
                for p1 in points_2d_set[i]:
                    for p2 in points_2d_set[j]:
                        points_candidate.append(
                            camera_system.find3d([(str(i), [p1[0], p1[1]]), (str(j), [p2[0], p2[1]])]))
                        all_2D.append(((str(i), [p1[0], p1[1]]), (str(j), [p2[0], p2[1]])))

    # compute the 3 nearest points
    tree = spt.cKDTree(data=points_candidate)
    min_distance = 1000000
    for n, point in enumerate(points_candidate):
        distances, indexs = tree.query(point, k=3)
        # print(distances[1], distances[2])
        if distances[1] + distances[2] < min_distance:
            min_distance = distances[1] + distances[2]
            center = np.mean([point, points_candidate[indexs[1]], points_candidate[indexs[2]]], axis=0)

    # choose camera
    points_use = []
    candidates = []
    left = []
    left_ID = []

    if min_distance > 2 * threshold:
        candidates.insert(0, center)
        return np.zeros((1, 3)), candidates

    for n, point in enumerate(points_candidate):
        if np.linalg.norm(point - center) < threshold:
            if all_2D[n][0] not in points_use:
                points_use.append(all_2D[n][0])
            if all_2D[n][1] not in points_use:
                points_use.append(all_2D[n][1])
        else:
            left.append(point)
            left_ID.append(n)
    points_3d = camera_system.find3d(points_use)

    # find the candidate(s)
    minn = 0
    while len(left) > 3:
        minn = 10000000
        tree_candidate = spt.cKDTree(data=left)
        for n, point in enumerate(left):
            distances, indexs = tree_candidate.query(point, k=3)
            if distances[1] + distances[2] < minn:
                minn = distances[1] + distances[2]
                center = np.mean([point, left[indexs[1]], left[indexs[2]]], axis=0)
        if minn < threshold:
            candidates_use = []
            tmp1 = []
            tmp2 = []
            for n, point in enumerate(left):
                if np.linalg.norm(point - center) < threshold:
                    if (all_2D[left_ID[n]][0] not in candidates_use):
                        candidates_use.append(all_2D[left_ID[n]][0])
                    if (all_2D[left_ID[n]][1] not in candidates_use):
                        candidates_use.append(all_2D[left_ID[n]][1])
                else:
                    tmp1.append(point)
                    tmp2.append(left_ID[n])
            left = tmp1
            left_ID = tmp2
            candidates.append(camera_system.find3d(candidates_use))
        else:
            break

    return points_3d, candidates


def estimate_ball_3d(x,imglist, camera_param, opt, model, device, imgsz=640):
    # device = select_device(opt.device)
    # print(f"in ball_3d, device = {device}")
    half = device != 'cpu'
    # model = attempt_load(weights, map_location=device)  # load FP32 model
    # if half:
    #     model.half()  # to FP16
    imgsz = check_img_size(int(str(imgsz)), s=model.stride.max())  # check img_size

    # 2D-Detection
    f = lambda x:process(x, imgsz)
    dataset = [f(i) for i in imglist]

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device != 'cpu' else None  # run once
    # print(_)
    # _ = _.detach().cpu()
    current_frame = {}
    # x = 0
    for n, (im0, img) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=opt.augment)[0]
        # pred = pred.detach().cpu().numpy()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                    # print(*xyxy)
                    # print(xyxy)
                    # plot_one_box(xyxy, im0, line_thickness=3)
                    # cv2.imwrite(f"{x}_{i}.jpg", im0)
                    # print(n+1, ((line[0] + line[2]) / 2, (line[1] + line[3]) / 2))
                    if n + 1 in current_frame:
                        current_frame[n + 1].append(((line[0] + line[2]) / 2, (line[1] + line[3]) / 2))
                    else:
                        current_frame[n + 1] = [((line[0] + line[2]) / 2, (line[1] + line[3]) / 2)]
        # print(current_frame[n+1])

        # txt_path =
        # if n == 1:
            # x = x + 1

        # with open(txt_path + '.txt', 'a') as f:
        #     f.write(('%g ' * 5 + '\n') % (count, *xyxy))  # label format
    # 3D-Reconstruction
    # print(current_frame)
    if len(current_frame.keys()) < 3:
        return None
    else:
        recon, candidates = triangulate_one_point_ransac(build_multi_camera_system(camera_param, no_distortion=False),
                                                         current_frame)
        if np.all(recon == 0):
            return None
        return recon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./model/best.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='', help='source')  # file/folder
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()