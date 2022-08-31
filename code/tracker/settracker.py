import numpy as np
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import cv2

from tracking_utils.utils import *
from tracking_utils.log import logger
from tracking_utils.kalman_filter import KalmanFilter
from tracker import matching
from .basetrack import BaseTrack, TrackState


class STrack(BaseTrack):
    def __init__(self, xyz, score, temp_feat, buffer_size=30):

        # wait activate
        self.joints = np.asarray(xyz, dtype=np.float)
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)  # view x 64
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_features(self, feat):
        views = feat.shape[0]
        for c in range(views):
            if np.sum(feat[c, :-1]) == 0:
                feat[c, :-1] += 1.0
        feat[:, :-1] /= np.linalg.norm(feat[:, :-1], axis=-1, keepdims=True)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat[:, :-1] = self.alpha * self.smooth_feat[:, :-1] + (1 - self.alpha) * feat[:, :-1]
            self.smooth_feat[:, -1] = feat[:, -1]
        self.features.append(feat)
        self.smooth_feat[:, :-1] /= np.linalg.norm(self.smooth_feat[:, :-1], axis=-1, keepdims=True)

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.state = TrackState.Tracked
        self.is_activated = True

        self.joints = new_track.joints
        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.det_thresh = 0.5
        self.buffer_size = int(frame_rate / 30.0 * 100)
        self.max_time_lost = self.buffer_size
        self.max_per_image = 50
        self.eta = 0

    def update(self, id_feature, dets):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(xyz[:, :3], xyz[0, 3], f, 30) for
                          (xyz, f) in zip(dets[:, :, :4], id_feature)]
        else:
            detections = []

        t2 = time.time()
        # print('Forward: {} s'.format(t2-t1))

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''

        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        dists_reid = matching.embedding_distance_set_weight(strack_pool, detections)
        #dists_coord = matching.coord_distance(strack_pool, detections, metric='euclidean')
        #dists_coord[dists_coord <= 2000] = 1.0
        #dists_coord[dists_coord > 2000] = np.inf
        #dists_reid = dists_reid * dists_coord
        dists_cos = matching.coord_distance(strack_pool, detections, metric='euclidean')
        dists_cos /= np.linalg.norm(dists_cos, axis=0, keepdims=True)
        dists_reid = self.eta * dists_cos + (1 - self.eta) * dists_reid
        #print(dists_reid)

        matches, u_track, u_detection = matching.linear_assignment(dists_reid, thresh=0.7)

        '''
        # use only 3D location
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        dists_coord = matching.coord_distance(strack_pool, detections, metric='euclidean')
        #dists_coord = matching.coord_distance(strack_pool, detections, metric='cosine')
        matches, u_track, u_detection = matching.linear_assignment(dists_coord, thresh=200)
        '''

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        #for it in u_track:
            #track = strack_pool[it]
            #if not track.state == TrackState.Lost:
                #track.mark_lost()
                #lost_stracks.append(track)

        ''' Step 3: Second association, with IOU'''
        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists_coord = matching.coord_distance(r_tracked_stracks, detections, metric='euclidean')
        matches, u_track, u_detection = matching.linear_assignment(dists_coord, thresh=200)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.coord_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=500)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        t4 = time.time()
        # print('Ramained match {} s'.format(t4-t3))
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]  # type: list[STrack]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        print(self.tracked_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        #logger.debug('===========Frame {}=========='.format(self.frame_id))
        #logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        #logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        #logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        #logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        t5 = time.time()
        # print('Final {} s'.format(t5-t4))
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.coord_distance(stracksa, stracksb, metric='euclidean')
    pairs = np.where(pdist < 10)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb