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


def slide_window(recon_list, pose_recon_list):
    # you shall set the threshold here, it determines the distance for a burst
    threshold_ball = 100
    threshold_pose = 100

    length = len(recon_list)
    dele = []
    cnt = 0 # counting the deleted points
    dictlist = []
    for key, value in recon_list.items():
        temp = (key,value)
        dictlist.append(temp)
    
    # a slide window in the size of 5
    print("into slide window for ball")
    for i in range(2,length-2):
        cur = dictlist[i]
        pre2 = dictlist[i-2]
        pre1 = dictlist[i-1]
        po2 = dictlist[i+2]
        po1 = dictlist[i+1]
        distpre = sqrt((pre2[1][0]-pre1[1][0])**2 + (pre2[1][1]-pre1[1][1])**2 + (pre2[1][2]-pre1[1][2])**2)
        dist1 = sqrt((pre1[1][0]-cur[1][0])**2 + (pre1[1][1]-cur[1][1])**2 + (cur[1][2]-pre1[1][2])**2)
        offset1 = abs(distpre - dist1)
        distpo = sqrt((po2[1][0]-po1[1][0])**2 + (po2[1][1]-po1[1][1])**2 + (po2[1][2]-po1[1][2])**2)
        dist2 = sqrt((cur[1][0]-po1[1][0])**2 + (cur[1][1]-po1[1][1])**2 + (cur[1][2]-po1[1][2])**2)
        offset2 = abs(distpo - dist2)
        
        # a burst
        if offset1 > threshold_ball and offset2 > threshold_ball:
            print('offset2 = %s, offset1 = %s, '% (offset2, offset1,))
            if cur[0] not in dele:
                dele.append(cur[0])
                cnt+=1
    for i in dele:
        recon_list.pop(i)
        pose_recon_list.pop(i)
    print("done slide window for ball")
    
    
    print("into slide window for pose")
    length = len(pose_recon_list)
    dele = []
    cnt = 0 # counting the deleted points
    dictlist = []
    for key, value in pose_recon_list.items():
        temp = (key,value) # key is frameid and value is (ids, poses)
        dictlist.append(temp)
    # print("dictlist goes like : \n{}".format(np.array(dictlist).shape))
    #initialize post and pre poses
    
    
    
    # for i in range(5):
    #     if i == 2:
    #         continue
    #     ids = dictlist[i][1][0]
    #     print("ids goes like {}".format(ids))
    #     poses = dictlist[i][1][1]
    #     print("poses for this frame looks like {}".format(np.array(poses).shape))
    #     for j in range(len(ids)):
    #         id = ids[j]
    #         if i == 0:
    #             pose_prev2[id] = poses[j][2]
    #         elif i == 1:
    #             pose_prev1[id] = poses[j][2]
    #         elif i == 3:
    #             pose_post1[id] = poses[j][2]
    #         elif i == 4:
    #             pose_post2[id] = poses[j][2]
        
    # a slide window in the size of 5
    for i in range(4,length-4):
        print(f"now the id is {i}")
        pose_prev4 = {}
        pose_post4 = {}
        pose_prev2 = {}
        pose_prev1 = {}
        pose_post1 = {}
        pose_post2 = {}
        # print("cleared cache\n")
        
        ids_prev4 = dictlist[i - 4][1][0]
        ids_post4 = dictlist[i + 4][1][0]
        # pose_prev4 = dictlist[i - 4][1][1]
        for j in range(4):
            if j == 0:
                t = -2
                ids_prev2 = dictlist[i + t][1][0]
            elif j == 1:
                t = -1
                ids_prev1 = dictlist[i + t][1][0]
            elif j == 2:
                t = 1
                ids_post1 = dictlist[i + t][1][0]
            else:
                t = 2
                ids_post2 = dictlist[i + t][1][0]
        # if True:
            ids_compare = dictlist[i + t][1][0]
            pose_compare = dictlist[i + t][1][1]
            # print(f"now the j is {j}")
            # print(f"pose compare goes like {np.array(pose_compare).shape}")
            # print(f"ids_compare goes like {ids_compare}")
            
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
        
        for j in ids: # delete a pop-up
            if j not in ids_prev4 and j not in ids_post4:
            # if j not in ids_post1 and j not in ids_post2 and j not in ids_prev1 and j not in ids_prev2:
                dele.append(frameid)
                
        for j in range(len(ids)): # length of ids
            id = ids[j]
            if id not in ids_post1 or id not in ids_post2 or id not in ids_prev1 or id not in ids_prev2:
                continue # not suitable for a slide window yet
            # id = ids[j]
            pose = cur_poses[j][2]
            pre2 = pose_prev2[id]
            pre1 = pose_prev1[id]
            po2 = pose_post1[id]
            po1 = pose_post1[id]
            
            distpre = sqrt((pre2[0]-pre1[0])**2 + (pre2[1]-pre1[1])**2 + (pre2[2]-pre1[2])**2)
            dist1 = sqrt((pre1[0]-pose[0])**2 + (pre1[1]-pose[1])**2 + (pose[2]-pre1[2])**2)
            offset1 = abs(distpre - dist1)
            distpo = sqrt((po2[0]-po1[0])**2 + (po2[1]-po1[1])**2 + (po2[2]-po1[2])**2)
            dist2 = sqrt((pose[0]-po1[0])**2 + (pose[1]-po1[1])**2 + (pose[2]-po1[2])**2)
            offset2 = abs(distpo - dist2)
            # print(f"offset1 = {offset1}, offset2 = {offset2}")
            # a burst
            if offset1 > threshold_pose and offset2 > threshold_pose:
                print('offset2 = %s, offset1 = %s, '% (offset2, offset1,))
                if frameid not in dele:
                    dele.append(frameid)
                    cnt+=1
            
            
    for i in dele:
        recon_list.pop(i)
        pose_recon_list.pop(i)
    
    
    
    # go through one euro filter
    print("into one-euro-filter for ball")
    dictlist = []
    for key, value in recon_list.items():
        temp = [key,value]
        dictlist.append(temp)

    #Fc and beta
    min_cutoff = 1
    beta = 0
    length = len(dictlist)
    x_hat = np.zeros((length,3),dtype=float)
    x_hat[0] = dictlist[0][1]
    one_euro_filter = OneEuroFilter(dictlist[0][0], dictlist[0][1],min_cutoff=min_cutoff,beta=beta)
    for i in range(1, length):
        x_hat[i] = one_euro_filter(dictlist[i][0], dictlist[i][1])
        # if needed, you could check the changes by printing them in the following way
        # print("ori: %s,%s,%s; filtered:%s,%s,%s" % (dictlist[i][1][0],dictlist[i][1][1],dictlist[i][1][2],x_hat[i][0],x_hat[i][1],x_hat[i][2]))
    
    for i in range(length):
        dictlist[i][1] = x_hat[i]
    new_recon = dict(dictlist)
    return new_recon, pose_recon_list