import numpy as np
import random

def farthest_point_sampling_fast(point_cloud, sample_num):
# farthest point sampling
# point_cloud: Nx3

    pc_num = len(point_cloud)

    if pc_num <= sample_num:
        sampled_idx = np.arange(pc_num).T
        sampled_idx = np.concatenate((sampled_idx, np.random.randint([1, pc_num], sample_num - pc_num, 1)), axis=0)
    else:
        sampled_idx = np.zeros(sample_num)
        sampled_idx[0] = np.random.randint([0, pc_num])

        cur_sample = np.tile(point_cloud[sampled_idx[0],:], (pc_num, 1))
        diff = point_cloud - cur_sample
        min_dist = np.dot(diff, diff)

        for cur_sample_idx in range(1, sample_num):
            # find the farthest point
            sampled_idx[cur_sample_idx] = max(min_dist)

            if cur_sample_idx < sample_num:
                # update min_dist
                valid_idx = (min_dist > 1e-8)
                diff = point_cloud[valid_idx, :] - np.tile(point_cloud[sampled_idx[cur_sample_idx], :], (sum(valid_idx), 1))
                min_dist[valid_idx, :] = min(min_dist[valid_idx, :], np.dot(diff, diff))
    sampled_idx = np.unique(sampled_idx)
    return sampled_idx