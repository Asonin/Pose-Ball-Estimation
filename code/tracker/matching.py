import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist
# from sklearn.utils import linear_assignment_
from scipy.optimize import linear_sum_assignment as linear_assignment
import torch

import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment_sklearn(cost_matrix, thresh):
    """
    Simple linear assignment
    :type cost_matrix: np.ndarray
    :type thresh: float
    :return: matches, unmatched_a, unmatched_b
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    cost_matrix[cost_matrix > thresh] = thresh + 1e-4
    indices = linear_assignment_.linear_assignment(cost_matrix)

    return _indices_to_matches(cost_matrix, indices, thresh)


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1, -1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    #print(cost_matrix)
    return cost_matrix


def embedding_distance_set(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)  # nd x view x 64
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1, -1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)  # nt x view x 64
    views = det_features.shape[1]
    dist_views = np.zeros((views, len(tracks), len(detections)), dtype=np.float)
    for c in range(views):
        dist_views[c] = cdist(track_features[:, c, :], det_features[:, c, :])
    dist_views[dist_views == 0] = 1.2
    dist_views_torch = torch.tensor(dist_views)
    dist_min = dist_views_torch.topk(5, dim=0, largest=False).values.mean(dim=0).cpu().numpy()
    #dist_min = np.mean(dist_views, axis=0)
    #print(dist_min)
    cost_matrix = np.maximum(0.0, dist_min)  # Nomalized features
    #print(cost_matrix)
    return cost_matrix


def embedding_distance_set_weight(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)  # nd x view x 65
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1, -1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)  # nt x view x 65
    views = det_features.shape[1]
    dist_views = np.zeros((views, len(tracks), len(detections)), dtype=np.float)
    weight_views = np.zeros((views, len(tracks), len(detections)), dtype=np.float)
    nt = track_features.shape[0]
    nd = det_features.shape[0]
    for c in range(views):
        dist_views[c] = cdist(track_features[:, c, :-1], det_features[:, c, :-1])   # nt x nd
        weight_views[c] = (1 - track_features[:, c, 64:65].repeat(nd, axis=1)) * (1 - det_features[:, c, 64:65].T.repeat(nt, axis=0))
    dist_views[dist_views == 0] = 1.2
    dist_views = dist_views * weight_views
    dist_views_torch = torch.tensor(dist_views)
    dist_min = dist_views_torch.topk(5, dim=0, largest=False).values.mean(dim=0).cpu().numpy()
    #dist_min = np.mean(dist_views, axis=0)
    #print(dist_min)
    cost_matrix = np.maximum(0.0, dist_min)  # Nomalized features
    #print(cost_matrix)
    return cost_matrix


def coord_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_coords = np.asarray([np.mean(track.joints[5:7], axis=0) for track in detections], dtype=np.float)
    for i, track in enumerate(tracks):
        cost_matrix[i, :] = np.maximum(0.0, cdist(np.mean(track.joints[5:7], axis=0).reshape(1, -1), det_coords, metric))
    return cost_matrix
