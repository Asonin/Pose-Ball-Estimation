# Extracted all frames of the videos
import os
from pathlib import Path
import parse
from skvideo.io import ffprobe, vreader
from skimage.io import imsave
from tqdm import tqdm
import pickle
import cv2
import numpy as np


if __name__ == '__main__':
    mp4_path = Path('../data/retail/')
    vfiles = sorted(list(mp4_path.glob('*.mp4')))
    nview = len(vfiles)

    with open(os.path.join(mp4_path, 'camera_calibration.pkl'), "rb") as f:
        camera_parameters = pickle.load(f)

    for cam_id, path in enumerate(vfiles):
        format_string = 'Cam_{:03d}.mp4'
        path_cam_id = parse.parse(format_string, path.name)[0] - 1
        assert cam_id == path_cam_id, 'Wrong match of camera and sequence'

        vid_frames = vreader(str(path.absolute()))

        # Note the id is 1 smaller than mp4
        output_path = mp4_path / f'Cam_{cam_id:03d}'
        if not output_path.exists():
            os.makedirs(output_path)

        vid_cam = camera_parameters[cam_id]
        m1, m2 = cv2.initUndistortRectifyMap(cameraMatrix=vid_cam['K'],
                                             distCoeffs=vid_cam['distortion'],
                                             R=np.eye(3, 3),
                                             newCameraMatrix=vid_cam['K'],
                                             size=(1920, 1080),
                                             m1type=cv2.CV_32FC1,
                                             )

        for idx, frame in enumerate(tqdm(vid_frames)):
            frame_out_path = output_path / f'{idx:08d}_undistorted.jpg'
            # frame_undistort = cv2.undistort(frame,
            # cameraMatrix=vid_cam['K'],
            # distCoeffs=vid_cam['distortion'],  # k1,k2,p1,p2,k3
            # )
            frame_undistort = cv2.remap(frame, m1, m2, cv2.INTER_CUBIC)

            imsave(frame_out_path, frame_undistort)
