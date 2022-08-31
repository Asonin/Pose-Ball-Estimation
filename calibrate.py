import subprocess
import signal
import os
import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import ipdb

def aruco_tracker(i, frame, mtx, dist, marker_len):
    flag = True
	# operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):
        flag = False
        # estimate pose of each marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, marker_len, mtx, dist)
        
        rotation_vector = rvec.reshape((3, 1))
        translation_vector = tvec.reshape((3, 1))

        rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)

        R = np.array(rotation_matrix)
        T = np.array(translation_vector) # * 100 # Convert length unit from m to cm

        # ipdb.set_trace()
        cam_pos = -np.matmul(rotation_matrix.T, T)
        print(i, cam_pos)
        # Save Camera Extrinsic Parameters in VoxelPose Style
        np.savez('params/' + i + '_extrinsic.npz', mtx, dist, R, cam_pos)
        #(rvec-tvec).any() # get rid of that nasty numpy value array error

        for j in range(0, ids.size):
            # draw axis for the aruco markers
            aruco.drawAxis(frame, mtx, dist, rvec[j], tvec[j], 50)
        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)

        # code to show ids of the marker found
        strg = ''
        for j in range(0, ids.size):
            strg += str(ids[j][0])+', '

        cv2.putText(frame, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        # code to show 'No Ids' when no markers are found
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    return flag, frame

addr_list = ["162.105.162.228", "162.105.162.144", "162.105.162.150", "162.105.162.103"]
name_list = ["228", "144", "150", "103"]
p_list = []
num_cams = len(addr_list)

# for i in range(num_cams):
#     addr, name = addr_list[i], name_list[i]
#     p = subprocess.Popen(["./calibrate %s %s &" % (addr, name)], shell=True)
#     p_list.append(p)


while True:
    try:
        for i in range(num_cams):
            addr, name = addr_list[i], name_list[i]
            cam =  np.load('params/' + name + '_intrinsic.npz')
            cam = dict(cam)
            # Camera Matrix
            mtx = cam['arr_0']
            # Distortion Coefficient
            dist = cam['arr_1']

            # Undistortion
            img = cv2.imread('out/%s.jpg' % name)
            h,  w = img.shape[: 2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

            # undistort
            undst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            undst = undst[y:y+h, x:x+w]
            # cv.imwrite(str(data_dir) + '_calibresult.jpg', dst)
            flag, result = aruco_tracker(name, undst, mtx, dist, 195)
            print(name, flag)
            cv2.imwrite('out/calibrated/%s.jpg' % name, result)
        print("\n")
        # pass
    except KeyboardInterrupt:
        for p in p_list:
            p.kill()
        print('All subprocesses killed')
        sys.exit()


