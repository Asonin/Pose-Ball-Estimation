import numpy as np
import cv2
from datetime import datetime
import pandas as pd
import random
import os
import argparse
import ipdb


parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default='test')
args = parser.parse_args()

addr_list = ["222.29.70.11",
             "222.29.70.12",
             "222.29.70.13",
             "222.29.70.14",
             "222.29.70.15",
             "222.29.70.16",
             "222.29.70.17",
             "222.29.70.18"]
cam_list = [str(x+1) for x in range(8)]
p_list = []


def within(idx, n_frames):
    for i in range(len(idx)):
        if idx[i]>=n_frames[i]:
            return False
    return True

def load_intrinsic(file_name):
    cam =  np.load(file_name)
    cam = dict(cam)
    # Camera Matrix
    mtx = cam['arr_0']
    # Distortion Coefficient
    dist = cam['arr_1']
    return mtx, dist


def undistort(img, mtx, dist):
    h,  w = img.shape[: 2]
#     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))
#     # undistort
#     undst = cv2.undistort(img, mtx, dist, None, newcameramtx)
#     # crop the image
# #     x, y, w, h = roi
# #     ipdb.set_trace().
#     return undst
    m1, m2 = cv2.initUndistortRectifyMap(cameraMatrix=mtx,
                                             distCoeffs=dist,
                                             R=np.eye(3, 3),
                                             newCameraMatrix=mtx,
                                             size=(w, h),
                                             m1type=cv2.CV_32FC1,
                                             )
    frame_undistort = cv2.remap(img, m1, m2, cv2.INTER_CUBIC)
    return frame_undistort
    
def export_combined(cam_list, matches):
    os.makedirs('out/%s/combined/' % args.prefix, exist_ok=True)
    for x, match in enumerate(matches):
        img_list = []
        for i, fid in enumerate(match):
            cam = cam_list[i]
            img = cv2.imread('out/%s/%s/%08d.jpg' % (args.prefix, cam, fid))
            img_list.append(img)
        combined_img = np.concatenate(img_list)
        cv2.imwrite('out/%s/result/%08d.jpg' % (args.prefix, x), combined_img)

def export_synced(cam_list, matches):
    os.makedirs('out/%s/synced/' % args.prefix, exist_ok=True)
    for cam in cam_list:
         os.makedirs('out/%s/synced/%s/' % (args.prefix, cam), exist_ok=True)
    for x, match in enumerate(matches):
        img_list = []
        for i, fid in enumerate(match):
            cam = cam_list[i]
            img = cv2.imread('out/%s/%s/%08d.jpg' % (args.prefix, cam, fid))
            cv2.imwrite('out/%s/synced/%s/%08d.jpg' % (args.prefix, cam, x), img)

num_cams = len(cam_list)


# Extract frames
for cam in cam_list:
    os.makedirs('out/%s/%s' % (args.prefix, cam), exist_ok=True)
    # if len(os.listdir('out/%s/%s' % (args.prefix, cam))) > 50:
    #     continue
    cap = cv2.VideoCapture('out/%s/%s.mp4' % (args.prefix, cam))
    i = 0
#     mtx, dist = load_intrinsic('params/' + cam + '_intrinsic.npz')
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            i += 1
#             frame = undistort(frame, mtx, dist)
            cv2.imwrite('out/%s/%s/%08d.jpg' % (args.prefix, cam, i), frame)
        else:
            break
    cap.release()
    print("Cam %s Done, %d frames." % (cam, i))
    
# Read Timestamps
ts = []
dts = []
n_frames = []
for i, cam in enumerate(cam_list):
    df=pd.read_csv('out/%s/%s.csv' % (args.prefix, cam), sep=',',header=None)
    cam_log = df.values
    t = []
    dt = []
    for idx, row in enumerate(cam_log):
        if idx<=1:
            continue
        t.append(int(row[2]))
        y, mo, d, h, mi, s, ms = int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7]), int(row[8]), int(row[9])
        dt.append(datetime(y, mo, d, h, mi, s, ms*1000).timestamp())                           # milisecond to microsecond
    frames = len(dt)
    dts.append(np.array(dt))
    ts.append(np.array(t))
    print("Cam %s Done, %d frames." % (cam, frames))
    n_frames.append(frames)


fps = 25.0
interval = 1/fps

idx = []
start = [dts[x][51] for x in range(num_cams)]
lff = max(start)           # Get the latest first frame.
ref = np.argmax(start)
for i in range(num_cams):
    idx.append(np.argmin(abs(dts[i]-lff)))
idx = np.array(idx)

matches = []
while within(idx, n_frames):
    for i in range(num_cams):
        while within(idx+1, n_frames) and (abs(dts[i][idx[i]+1]-dts[ref][idx[ref]]) < abs(dts[i][idx[i]]-dts[ref][idx[ref]])):
            idx[i] += 1
        while within(idx-1, n_frames) and (abs(dts[i][idx[i]-1]-dts[ref][idx[ref]]) < abs(dts[i][idx[i]]-dts[ref][idx[ref]])):
            idx[i] -= 1
    matches.append(idx.copy())
    idx += 1

matches = np.array(matches)

print("Matched sequence length: %d" % len(matches))

diff = []
for i in range(num_cams):
    for j in range(i): 
        if i==j:
            continue
        diff.append(abs(dts[i][matches[:,i]]-dts[j][matches[:,j]]))

print("Max Diff = %.5fs, Mean Diff = %.5fs" % (np.max(diff), np.mean(diff)))

# export_combined(cam_list, matches)
export_synced(cam_list, matches)