from cmath import sqrt
from operator import truediv

import numpy as np
import math

def smoothing_factor(t_e, cutoff):
    t_e = np.array(t_e)
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)


def exponential_smoothing(a, x, x_prev):
    a = np.array(a)
    x = np.array(x)
    x_prev = np.array(x_prev)
    return a * x + (1 - a) * x_prev


class OneEuroFilter:
    def __init__(self, t0, x0, dx0=(0.0,0.0,0.0), min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        # print(x)
        # print(self.x_prev)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        # offset = x_hat - x
        

        return x_hat


class OneEuroFilterPose(OneEuroFilter):
    def __init__(self, t0, x0, ids, dx0=(0.0,0.0,0.0), min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.ids = ids

        # Previous values.
        self.x_prev = np.zeros((50,3))
        self.dx_prev = np.zeros((50,3))
        self.t_prev = np.zeros((50,3))
        # print(x0)
        for i in range(len(ids)):
            id = ids[i]
            self.x_prev[id] = x0[i][2]
            self.dx_prev[id] = dx0
            self.t_prev[id] = t0

    def __call__(self, t, x, ids):
        """Compute the filtered signal."""
        # do it for every person
        for i in range(len(ids)):
            id = ids[i]
            if id not in self.ids:
                print("new id: %d" % id)
                self.dx_prev[id] = (0,0,0)
                self.x_prev[id] = x[i][2]
                self.t_prev[id] = t
                self.ids.append(id)
                print("now the ids are: %s" % np.array(self.ids))
                continue

            t_e = t - self.t_prev[id]

            # The filtered derivative of the signal.
            a_d = smoothing_factor(t_e, self.d_cutoff)
            dx = (x[i][2] - self.x_prev[id]) / t_e
            dx_hat = exponential_smoothing(a_d, dx, self.dx_prev[id])

            # The filtered signal.
            cutoff = self.min_cutoff + self.beta * abs(dx_hat)
            a = smoothing_factor(t_e, cutoff)
            x_hat = exponential_smoothing(a, x[i][2], self.x_prev[id])

            # Memorize the previous values.
            self.x_prev[id] = x_hat
            self.dx_prev[id] = dx_hat
            self.t_prev[id] = t
            # print("ori: %s,%s,%s; filtered:%s,%s,%s" % (x[i][2][0],x[i][2][1],x[i][2][2],x_hat[0],x_hat[1],x_hat[2]))
            offset = x_hat - x[i][2]
            # print("offset = %s,%s,%s" % (offset[0], offset[1], offset[2]))
            for j in range(15):
                x[i][j] = x[i][j] + offset

        return x


def do_interpolation(recon_list, pose_list):
    print("into interpolation")
    recon_list = dict(recon_list)
    pose_list = dict(pose_list)
    recon = {}
    pose = {}
    
    i = iter(recon_list)
    t = len(recon_list)
    while t:
        key = next(i)
        t -= 1
        recon[key] = recon_list[key]
        
        tmp = pose_list[key]
        tmp = dict(zip(tmp[0],tmp[1]))
        tmp = dict(sorted(tmp.items(), key = lambda x:x[0]))
        pose0 = [list(tmp.keys()), list(tmp.values())]
        # print(pose0[0])
        pose[key] = pose0
        
        if t == 0:
            break
        key1 = next(i)
        t -= 1
        # key1 = int(key1)
        # key = int(key)
        offset = key1 - key
        # print(f"offset for index is {offset}")
        tmp1 = pose_list[key1]
        tmp1 = dict(zip(tmp1[0],tmp1[1]))
        tmp1 = dict(sorted(tmp1.items(), key = lambda x:x[0])) 
        pose1 = [list(tmp1.keys()), list(tmp1.values())]
        # print(pose1[0])
        
        if offset >= 10 or offset == 1:
            recon[key1] = recon_list[key1]
            pose[key1] = pose1
            continue 
        else: # do interpolation
            print(f"doing interpolation for {offset}")
            ball_pos0 = recon_list[key]
            ball_pos1 = recon_list[key1]
            ball_off = np.array(ball_pos1) - np.array(ball_pos0)
            ball_off = ball_off / offset
            
            pose0 = np.array(pose[key][1])
            pose1 = np.array(pose1[1])
            pose_off = pose1 - pose0
            pose_off = pose_off / offset
            
            for j in range(1,offset):
                if j == 1:
                    recon[key + j] = recon[key] + ball_off
                    posej = pose0 + pose_off
                    pose[key + j] = [pose[key][0], posej]
                else:
                    recon[key + j] = recon[key + j - 1] + ball_off
                    posej = pose[key + j - 1][1] + pose_off
                    pose[key + j] = [pose[key + j - 1][0], posej]
    # for key, value in recon.items():
    #     print(key, value)
    # print("printing poses")    
    # for key,value in pose.items():
    #     print(key,value)            
    return recon, pose
        

def slide_window(recon_list, pose_recon_list):
    # you shall set the threshold here, it determines the distance for a burst
    threshold_ball = 500
    threshold_pose = 400

    length = len(recon_list)
    dele = []
    cnt = 0 # counting the deleted points
    dictlist = []
    for key, value in recon_list.items():
        temp = (key,value)
        dictlist.append(temp)
    # print(f"len of ball is {len(dictlist)}")
    # print(length)
    # print(len(pose_recon_list))
    # a slide window in the size of 5
    print("into slide window for ball")
    for i in range(1,length-1):
        cur = dictlist[i]
        # pre2 = dictlist[i-2]
        pre1 = dictlist[i-1]
        # po2 = dictlist[i+2]
        po1 = dictlist[i+1]
        mid = (np.array(pre1[1]) + np.array(po1[1]))/2
        dist1 = abs(sqrt((mid[0]-cur[1][0])**2 + (mid[1]-cur[1][1])**2 + (mid[2]-cur[1][2])**2))
        
        # distpre = sqrt((pre2[1][0]-pre1[1][0])**2 + (pre2[1][1]-pre1[1][1])**2 + (pre2[1][2]-pre1[1][2])**2)
        # dist1 = abs(sqrt((pre1[1][0]-cur[1][0])**2 + (pre1[1][1]-cur[1][1])**2 + (cur[1][2]-pre1[1][2])**2))
        print(dist1)
        # offset1 = abs(distpre - dist1)
        # distpo = sqrt((po2[1][0]-po1[1][0])**2 + (po2[1][1]-po1[1][1])**2 + (po2[1][2]-po1[1][2])**2)
        # dist2 = sqrt((cur[1][0]-po1[1][0])**2 + (cur[1][1]-po1[1][1])**2 + (cur[1][2]-po1[1][2])**2)
        # offset2 = abs(distpo - dist2)
        
        # a burst
        if int(dist1) > threshold_ball:
            if cur[0] not in dele:
                dele.append(cur[0])
                cnt+=1
        # if offset1 > threshold_ball and offset2 > threshold_ball:
        #     # print('offset2 = %s, offset1 = %s, '% (offset2, offset1,))
        #     if cur[0] not in dele:
        #         dele.append(cur[0])
        #         cnt+=1
    # for i in dele:
    #     recon_list.pop(i)
    #     pose_recon_list.pop(i)
    print(f'deleted {cnt} frames')
    print("done slide window for ball")
    
    
    print("into slide window for pose")
    length = len(pose_recon_list)
    # dele = []
    # cnt = 0 # counting the deleted points
    dictlist = []
    for key, value in pose_recon_list.items():
        temp = (key,value) # key is frameid and value is (ids, poses)
        dictlist.append(temp)
    # print("dictlist goes like : \n{}".format(np.array(dictlist).shape))
    #initialize post and pre poses
        
    # a slide window in the size of 9
    for i in range(2,length-2):
    # for i in range(4,length-4):
        # print(f"now the id is {i}")
        # print(length)
        # print(len(dictlist))
        # print(len(recon_list))
        pose_prev2 = {}
        pose_prev1 = {}
        pose_post1 = {}
        pose_post2 = {}
        
        # ids_prev4 = dictlist[i - 4][1][0]
        # ids_post4 = dictlist[i + 4][1][0]
        # pose_prev4 = dictlist[i - 4][1][1]
        for j in range(4):
            if j == 0:
                t = -2
                ids_prev2 = dictlist[i + t][1][0]
                # if len(ids_prev2) != num_people:
                #     dele.append(2)
            elif j == 1:
                t = -1
                ids_prev1 = dictlist[i + t][1][0]
            elif j == 2:
                t = 1
                ids_post1 = dictlist[i + t][1][0]
            else:
                t = 2
                ids_post2 = dictlist[i + t][1][0]
            ids_compare = dictlist[i + t][1][0]
            pose_compare = dictlist[i + t][1][1]
            # print(f"now the j is {j}")
            # print(f"pose compare goes like {np.array(pose_compare).shape}")
            # print(f"ids_compare goes like {ids_compare}")
            
            # set previous ids
            for k in range(len(pose_compare)):
                id = ids_compare[k]
                if j == 0:
                    pose_prev2[id] = pose_compare[k][2]
                elif j == 1:
                    pose_prev1[id] = pose_compare[k][2]
                elif j == 2:
                    pose_post1[id] = pose_compare[k][2]
                elif j == 3:
                    pose_post2[id] = pose_compare[k][2]
        
        frameid = dictlist[i][0]
        ids = dictlist[i][1][0]
        cur_poses = dictlist[i][1][1]
        flag = False
        for j in ids: # delete a pop-up
            # if j not in ids_prev4 and j not in ids_post4:
            # # if j not in ids_post1 and j not in ids_post2 and j not in ids_prev1 and j not in ids_prev2:
            #     # dele.append(frameid)
            #     flag = True
            #     break
            if j not in ids_post1 or j not in ids_post2 or j not in ids_prev1 or j not in ids_prev2:
                flag = True
                break
        if flag:
            # dele.append(frameid)
            continue
        
        for j in range(len(ids)): # length of ids
            id = ids[j]
            pose = cur_poses[j][2]
            # pre2 = pose_prev2[id]
            pre1 = pose_prev1[id]
            # po2 = pose_post1[id]
            po1 = pose_post1[id]
            
            mid = (np.array(po1) + np.array(pre1))/2
            dist1 = abs(sqrt((mid[0]-pose[0])**2 + (mid[1]-pose[1])**2 + (mid[2]-pre1[2])**2))
            
            # distpre = sqrt((pre2[0]-pre1[0])**2 + (pre2[1]-pre1[1])**2 + (pre2[2]-pre1[2])**2)
            # dist1 = abs(sqrt((pre1[0]-pose[0])**2 + (pre1[1]-pose[1])**2 + (pose[2]-pre1[2])**2))
            print(dist1)
            # offset1 = abs(distpre - dist1)
            # distpo = sqrt((po2[0]-po1[0])**2 + (po2[1]-po1[1])**2 + (po2[2]-po1[2])**2)
            # dist2 = sqrt((pose[0]-po1[0])**2 + (pose[1]-po1[1])**2 + (pose[2]-po1[2])**2)
            # offset2 = abs(distpo - dist2)
            # print(f"offset1 = {offset1}, offset2 = {offset2}")
            # a burst
            if int(dist1) > threshold_pose: 
            # if offset1 > threshold_pose and offset2 > threshold_pose:
                # print('offset2 = %s, offset1 = %s, '% (offset2, offset1,))
                if frameid not in dele:
                    dele.append(frameid)
                    cnt+=1  
    for i in dele: 
        recon_list.pop(i)
        pose_recon_list.pop(i)
    print(f'deleted {cnt} frames')
    
    # go through one euro filter
    # print("into one-euro-filter for ball")
    # dictlist = []
    # for key, value in recon_list.items():
    #     temp = [key,value]
    #     dictlist.append(temp)
    # #Fc and beta
    # min_cutoff = 1
    # beta = 0
    # length = len(dictlist)
    # x_hat = np.zeros((length,3),dtype=float)
    # x_hat[0] = dictlist[0][1]
    # one_euro_filter = OneEuroFilter(dictlist[0][0], dictlist[0][1],min_cutoff=min_cutoff,beta=beta)
    # for i in range(1, length):
    #     x_hat[i] = one_euro_filter(dictlist[i][0], dictlist[i][1])
    #     # if needed, you could check the changes by printing them in the following way
    #     # print("ori: %s,%s,%s; filtered:%s,%s,%s" % (dictlist[i][1][0],dictlist[i][1][1],dictlist[i][1][2],x_hat[i][0],x_hat[i][1],x_hat[i][2]))
    
    # for i in range(length):
    #     dictlist[i][1] = x_hat[i]
    # new_recon = dict(dictlist)
    return recon_list, pose_recon_list
