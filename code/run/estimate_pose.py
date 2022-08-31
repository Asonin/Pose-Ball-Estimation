# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import json_tricks as json
import cv2

import run._init_paths
from lib.utils.transforms import get_affine_transform, get_scale
import pose_models

def load_test_model(config, test_model_file):
    gpus = [int(i) for i in config.GPUS.split(',')]
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    print('=> Constructing models...')
    model = eval('pose_models.' + config.MODEL + '.get')(config, is_train=False)
    with torch.no_grad():
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpus)
        model.to(f'cuda:{model.device_ids[0]}')

    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        print('=> load models state {}'.format(test_model_file))
        print('--------------------------------------')
        print(f"in load_model, model.device_ids[0] = {model.device_ids[0]}")
        model.module.load_state_dict(torch.load(test_model_file, map_location=f'cuda:{model.device_ids[0]}'))
    else:
        raise ValueError('Check the model file for testing!')

    print("=> Successfully load the model for 3D pose estimation...")
    model.eval()
    return model

def get_cam(cam_file):
    with open(cam_file) as cfile:
        cameras = json.load(cfile)

    cameras_int = {}
    for id, cam in cameras.items():
        cameras_int[int(id)] = cam
    cameras = cameras_int

    for id, cam in cameras.items():
        for k, v in cam.items():
            cameras[id][k] = np.array(v)

    return cameras

def estimate_pose_3d(config, model, images, meta):
    model.eval()
    with torch.no_grad():
        final_poses, poses, _, _, input_heatmap = model(views=images, meta=meta)
        final_poses = final_poses.detach().cpu().numpy()
        poses = poses.detach().cpu().numpy()
        input_heatmap = [heatmap.detach().cpu().numpy() for heatmap in input_heatmap]
        return final_poses, poses, input_heatmap

def get_params(config):
    ori_image_width = config.DATASET.ORI_IMAGE_WIDTH
    ori_image_height = config.DATASET.ORI_IMAGE_HEIGHT
    image_size = config.NETWORK.IMAGE_SIZE

    r = 0
    c = np.array([ori_image_width / 2.0, ori_image_height / 2.0])
    s = get_scale((ori_image_width, ori_image_height), image_size)
    resize_transform = get_affine_transform(c, s, r, image_size)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.ToTensor(), normalize])

    meta = {'seq': ['test']}
    return resize_transform, transform, meta