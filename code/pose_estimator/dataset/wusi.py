# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import json_tricks as json
import logging
import cv2

from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)

panoptic_joints_def = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
}

panoptic_bones_def = [
    [0, 1], [0, 2],  # trunk
    [0, 3], [3, 4], [4, 5],  # left arm
    [0, 9], [9, 10], [10, 11],  # right arm
    [2, 6], [6, 7], [7, 8],  # left leg
    [2, 12], [12, 13], [13, 14],  # right leg
]

class Wusi(JointsDataset):
    def __init__(self, cfg, is_train=True, transform=None):
        super().__init__(cfg, is_train, transform)
        self.has_evaluate_function = False
        self.frame_range = list(range(10)) 
        self.num_joints = len(panoptic_joints_def)
        self.cameras = self._get_cam()
        self._get_db()

    def _get_db(self):
        for i in self.frame_range:
            all_image_path = []
            missing_image = False
            for k in range(self.num_views):
                image_path = osp.join(str(k+1), "{:08d}.jpg".format(i))
                if not osp.exists(osp.join(self.dataset_root, image_path)):
                    logger.info("Image not found: {}. Skipped.".format(image_path))
                    missing_image = True
                    break
                data_numpy = cv2.imread(osp.join(self.dataset_root, image_path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

                # resize the image for preprocessing
                if data_numpy.shape[0] == self.ori_image_height:
                    input = cv2.warpAffine(data_numpy, self.resize_transform, 
                            (int(self.image_size[0]), int(self.image_size[1])),
                            flags=cv2.INTER_LINEAR)
                    cv2.imwrite(osp.join(self.dataset_root, image_path), input)
                    print("resize and overwrite the image:", image_path)

                all_image_path.append(osp.join(self.dataset_root, image_path))

            if missing_image:
                continue
            
            self.db.append({
                'seq': 'wusi',
                'all_image_path': all_image_path
            })

        super()._rebuild_db()
        logger.info("=> {} images from {} views loaded".format(len(self.db), self.num_views))
        return

    def _get_cam(self):
        cam_file = osp.join(self.dataset_root, 'calibration_wusi.json')
        with open(cam_file) as cfile:
            cameras = json.load(cfile)

        for id, cam in cameras.items():
            for k, v in cam.items():
                cameras[id][k] = np.array(v)
        
        cameras_int_key = {}
        for id, cam in cameras.items():
            cameras_int_key[int(id)] = cam

        our_cameras = dict()
        our_cameras['wusi'] = cameras_int_key
        return our_cameras

    def __getitem__(self, idx):
        input, target, meta, input_heatmap = super().__getitem__(idx)
        return input, target, meta, input_heatmap

    def __len__(self):
        return len(self.db)