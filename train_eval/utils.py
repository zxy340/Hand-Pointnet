'''
utils
author: Liuhao Ge
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def group_points(points, opt):
    # group points using knn and ball query
    # points: B * 1024 * 6
    cur_train_size = len(points)
    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM) \
                 - points[:,0:opt.sample_num_level1,0:3].unsqueeze(-1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM)# B * 512 * 3 * 1024
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 512 * 3 * 1024
    inputs1_diff = inputs1_diff.sum(2)                      # B * 512 * 1024
    dists, inputs1_idx = torch.topk(inputs1_diff, opt.knn_K, 2, largest=False, sorted=False)  # dists: B * 512 * 64; inputs1_idx: B * 512 * 64
        
    # ball query
    invalid_map = dists.gt(opt.ball_radius) # B * 512 * 64
    for jj in range(opt.sample_num_level1):
        inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = inputs1_idx.view(cur_train_size,opt.sample_num_level1*opt.knn_K,1).expand(cur_train_size,opt.sample_num_level1*opt.knn_K,opt.INPUT_FEATURE_NUM)
    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size,opt.sample_num_level1,opt.knn_K,opt.INPUT_FEATURE_NUM) # B*512*64*6

    inputs_level1_center = points[:,0:opt.sample_num_level1,0:3].unsqueeze(2)       # B*512*1*3
    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center.expand(cur_train_size,opt.sample_num_level1,opt.knn_K,3)
    inputs_level1 = inputs_level1.unsqueeze(1).transpose(1,4).squeeze(4)  # B*6*512*64
    inputs_level1_center = inputs_level1_center.contiguous().view(-1,1,opt.sample_num_level1,3).transpose(1,3)  # B*3*512*1
    return inputs_level1, inputs_level1_center
    #inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1
    
def group_points_2(points, sample_num_level1, sample_num_level2, knn_K, ball_radius):
    # group points using knn and ball query
    # points: B*(3+128)*512
    cur_train_size = points.size(0)
    inputs1_diff = points[:,0:3,:].unsqueeze(1).expand(cur_train_size,sample_num_level2,3,sample_num_level1) \
                 - points[:,0:3,0:sample_num_level2].transpose(1,2).unsqueeze(-1).expand(cur_train_size,sample_num_level2,3,sample_num_level1)# B * 128 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512
    dists, inputs1_idx = torch.topk(inputs1_diff, knn_K, 2, largest=False, sorted=False)  # dists: B * 128 * 64; inputs1_idx: B * 128 * 64
        
    # ball query
    invalid_map = dists.gt(ball_radius) # B * 128 * 64, invalid_map.float().sum()
    #pdb.set_trace()
    for jj in range(sample_num_level2):
        inputs1_idx.data[:,jj,:][invalid_map.data[:,jj,:]] = jj

    idx_group_l1_long = inputs1_idx.view(cur_train_size,1,sample_num_level2*knn_K).expand(cur_train_size,points.size(1),sample_num_level2*knn_K)
    inputs_level2 = points.gather(2,idx_group_l1_long).view(cur_train_size,points.size(1),sample_num_level2,knn_K) # B*131*128*64

    inputs_level2_center = points[:,0:3,0:sample_num_level2].unsqueeze(3)       # B*3*128*1
    inputs_level2[:,0:3,:,:] = inputs_level2[:,0:3,:,:] - inputs_level2_center.expand(cur_train_size,3,sample_num_level2,knn_K) # B*3*128*64
    return inputs_level2, inputs_level2_center
    # inputs_level2: B*131*sample_num_level2*knn_K, inputs_level2_center: B*3*sample_num_level2*1

def handjointsplot(outputs_xyz, gt_xyz, images):
    index = np.arange(21)
    gt_x_01 = gt_xyz[0, index[[0, 1]] * 3]
    gt_y_01 = gt_xyz[0, index[[0, 1]] * 3 + 1]
    gt_z_01 = gt_xyz[0, index[[0, 1]] * 3 + 2]
    gt_x_01 = gt_x_01.cpu().numpy()
    gt_y_01 = gt_y_01.cpu().numpy()
    gt_z_01 = gt_z_01.cpu().numpy()
    gt_x_02 = gt_xyz[0, index[[0, 5]] * 3]
    gt_y_02 = gt_xyz[0, index[[0, 5]] * 3 + 1]
    gt_z_02 = gt_xyz[0, index[[0, 5]] * 3 + 2]
    gt_x_02 = gt_x_02.cpu().numpy()
    gt_y_02 = gt_y_02.cpu().numpy()
    gt_z_02 = gt_z_02.cpu().numpy()
    gt_x_03 = gt_xyz[0, index[[0, 9]] * 3]
    gt_y_03 = gt_xyz[0, index[[0, 9]] * 3 + 1]
    gt_z_03 = gt_xyz[0, index[[0, 9]] * 3 + 2]
    gt_x_03 = gt_x_03.cpu().numpy()
    gt_y_03 = gt_y_03.cpu().numpy()
    gt_z_03 = gt_z_03.cpu().numpy()
    gt_x_04 = gt_xyz[0, index[[0, 13]] * 3]
    gt_y_04 = gt_xyz[0, index[[0, 13]] * 3 + 1]
    gt_z_04 = gt_xyz[0, index[[0, 13]] * 3 + 2]
    gt_x_04 = gt_x_04.cpu().numpy()
    gt_y_04 = gt_y_04.cpu().numpy()
    gt_z_04 = gt_z_04.cpu().numpy()
    gt_x_05 = gt_xyz[0, index[[0, 17]] * 3]
    gt_y_05 = gt_xyz[0, index[[0, 17]] * 3 + 1]
    gt_z_05 = gt_xyz[0, index[[0, 17]] * 3 + 2]
    gt_x_05 = gt_x_05.cpu().numpy()
    gt_y_05 = gt_y_05.cpu().numpy()
    gt_z_05 = gt_z_05.cpu().numpy()
    gt_x_1 = gt_xyz[0, index[1:5] * 3]
    gt_y_1 = gt_xyz[0, index[1:5] * 3 + 1]
    gt_z_1 = gt_xyz[0, index[1:5] * 3 + 2]
    gt_x_1 = gt_x_1.cpu().numpy()
    gt_y_1 = gt_y_1.cpu().numpy()
    gt_z_1 = gt_z_1.cpu().numpy()
    gt_x_2 = gt_xyz[0, index[5:9] * 3]
    gt_y_2 = gt_xyz[0, index[5:9] * 3 + 1]
    gt_z_2 = gt_xyz[0, index[5:9] * 3 + 2]
    gt_x_2 = gt_x_2.cpu().numpy()
    gt_y_2 = gt_y_2.cpu().numpy()
    gt_z_2 = gt_z_2.cpu().numpy()
    gt_x_3 = gt_xyz[0, index[9:13] * 3]
    gt_y_3 = gt_xyz[0, index[9:13] * 3 + 1]
    gt_z_3 = gt_xyz[0, index[9:13] * 3 + 2]
    gt_x_3 = gt_x_3.cpu().numpy()
    gt_y_3 = gt_y_3.cpu().numpy()
    gt_z_3 = gt_z_3.cpu().numpy()
    gt_x_4 = gt_xyz[0, index[13:17] * 3]
    gt_y_4 = gt_xyz[0, index[13:17] * 3 + 1]
    gt_z_4 = gt_xyz[0, index[13:17] * 3 + 2]
    gt_x_4 = gt_x_4.cpu().numpy()
    gt_y_4 = gt_y_4.cpu().numpy()
    gt_z_4 = gt_z_4.cpu().numpy()
    gt_x_5 = gt_xyz[0, index[17:21] * 3]
    gt_y_5 = gt_xyz[0, index[17:21] * 3 + 1]
    gt_z_5 = gt_xyz[0, index[17:21] * 3 + 2]
    gt_x_5 = gt_x_5.cpu().numpy()
    gt_y_5 = gt_y_5.cpu().numpy()
    gt_z_5 = gt_z_5.cpu().numpy()

    outputs_x_01 = outputs_xyz[0, index[[0, 1]] * 3]
    outputs_y_01 = outputs_xyz[0, index[[0, 1]] * 3 + 1]
    outputs_z_01 = outputs_xyz[0, index[[0, 1]] * 3 + 2]
    outputs_x_01 = outputs_x_01.cpu().numpy()
    outputs_y_01 = outputs_y_01.cpu().numpy()
    outputs_z_01 = outputs_z_01.cpu().numpy()
    outputs_x_02 = outputs_xyz[0, index[[0, 5]] * 3]
    outputs_y_02 = outputs_xyz[0, index[[0, 5]] * 3 + 1]
    outputs_z_02 = outputs_xyz[0, index[[0, 5]] * 3 + 2]
    outputs_x_02 = outputs_x_02.cpu().numpy()
    outputs_y_02 = outputs_y_02.cpu().numpy()
    outputs_z_02 = outputs_z_02.cpu().numpy()
    outputs_x_03 = outputs_xyz[0, index[[0, 9]] * 3]
    outputs_y_03 = outputs_xyz[0, index[[0, 9]] * 3 + 1]
    outputs_z_03 = outputs_xyz[0, index[[0, 9]] * 3 + 2]
    outputs_x_03 = outputs_x_03.cpu().numpy()
    outputs_y_03 = outputs_y_03.cpu().numpy()
    outputs_z_03 = outputs_z_03.cpu().numpy()
    outputs_x_04 = outputs_xyz[0, index[[0, 13]] * 3]
    outputs_y_04 = outputs_xyz[0, index[[0, 13]] * 3 + 1]
    outputs_z_04 = outputs_xyz[0, index[[0, 13]] * 3 + 2]
    outputs_x_04 = outputs_x_04.cpu().numpy()
    outputs_y_04 = outputs_y_04.cpu().numpy()
    outputs_z_04 = outputs_z_04.cpu().numpy()
    outputs_x_05 = outputs_xyz[0, index[[0, 17]] * 3]
    outputs_y_05 = outputs_xyz[0, index[[0, 17]] * 3 + 1]
    outputs_z_05 = outputs_xyz[0, index[[0, 17]] * 3 + 2]
    outputs_x_05 = outputs_x_05.cpu().numpy()
    outputs_y_05 = outputs_y_05.cpu().numpy()
    outputs_z_05 = outputs_z_05.cpu().numpy()
    outputs_x_1 = outputs_xyz[0, index[1:5] * 3]
    outputs_y_1 = outputs_xyz[0, index[1:5] * 3 + 1]
    outputs_z_1 = outputs_xyz[0, index[1:5] * 3 + 2]
    outputs_x_1 = outputs_x_1.cpu().numpy()
    outputs_y_1 = outputs_y_1.cpu().numpy()
    outputs_z_1 = outputs_z_1.cpu().numpy()
    outputs_x_2 = outputs_xyz[0, index[5:9] * 3]
    outputs_y_2 = outputs_xyz[0, index[5:9] * 3 + 1]
    outputs_z_2 = outputs_xyz[0, index[5:9] * 3 + 2]
    outputs_x_2 = outputs_x_2.cpu().numpy()
    outputs_y_2 = outputs_y_2.cpu().numpy()
    outputs_z_2 = outputs_z_2.cpu().numpy()
    outputs_x_3 = outputs_xyz[0, index[9:13] * 3]
    outputs_y_3 = outputs_xyz[0, index[9:13] * 3 + 1]
    outputs_z_3 = outputs_xyz[0, index[9:13] * 3 + 2]
    outputs_x_3 = outputs_x_3.cpu().numpy()
    outputs_y_3 = outputs_y_3.cpu().numpy()
    outputs_z_3 = outputs_z_3.cpu().numpy()
    outputs_x_4 = outputs_xyz[0, index[13:17] * 3]
    outputs_y_4 = outputs_xyz[0, index[13:17] * 3 + 1]
    outputs_z_4 = outputs_xyz[0, index[13:17] * 3 + 2]
    outputs_x_4 = outputs_x_4.cpu().numpy()
    outputs_y_4 = outputs_y_4.cpu().numpy()
    outputs_z_4 = outputs_z_4.cpu().numpy()
    outputs_x_5 = outputs_xyz[0, index[17:21] * 3]
    outputs_y_5 = outputs_xyz[0, index[17:21] * 3 + 1]
    outputs_z_5 = outputs_xyz[0, index[17:21] * 3 + 2]
    outputs_x_5 = outputs_x_5.cpu().numpy()
    outputs_y_5 = outputs_y_5.cpu().numpy()
    outputs_z_5 = outputs_z_5.cpu().numpy()

    fig1 = plt.figure()
    plt.imshow(images[0])
    plt.title('depth image')
    fig2 = plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot(gt_x_01, gt_y_01, gt_z_01, 'g.-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_02, gt_y_02, gt_z_02, 'b.-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_03, gt_y_03, gt_z_03, 'k.-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_04, gt_y_04, gt_z_04, 'r.-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_05, gt_y_05, gt_z_05, 'y.-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_1, gt_y_1, gt_z_1, 'g.-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_2, gt_y_2, gt_z_2, 'b.-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_3, gt_y_3, gt_z_3, 'k.-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_4, gt_y_4, gt_z_4, 'r.-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_5, gt_y_5, gt_z_5, 'y.-', linewidth=1, alpha=0.6)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.title('GT')
    fig3 = plt.figure()
    ax = plt.gca(projection='3d')
    ax.plot(outputs_x_01, outputs_y_01, outputs_z_01, 'g.-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_02, outputs_y_02, outputs_z_02, 'b.-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_03, outputs_y_03, outputs_z_03, 'k.-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_04, outputs_y_04, outputs_z_04, 'r.-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_05, outputs_y_05, outputs_z_05, 'y.-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_1, outputs_y_1, outputs_z_1, 'g.-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_2, outputs_y_2, outputs_z_2, 'b.-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_3, outputs_y_3, outputs_z_3, 'k.-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_4, outputs_y_4, outputs_z_4, 'r.-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_5, outputs_y_5, outputs_z_5, 'y.-', linewidth=1, alpha=0.6)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.title('Prediction')
    plt.show()

def handplot(points, outputs_xyz, gt_xyz, images):
    points = points.cpu().numpy()
    x = points[0, :, 0]
    y = points[0, :, 1]
    z = points[0, :, 2]

    index = np.arange(21)
    gt_x_01 = gt_xyz[0, index[[0, 1]] * 3]
    gt_y_01 = gt_xyz[0, index[[0, 1]] * 3 + 1]
    gt_z_01 = gt_xyz[0, index[[0, 1]] * 3 + 2]
    gt_x_01 = gt_x_01.cpu().numpy()
    gt_y_01 = gt_y_01.cpu().numpy()
    gt_z_01 = gt_z_01.cpu().numpy()
    gt_x_02 = gt_xyz[0, index[[0, 5]] * 3]
    gt_y_02 = gt_xyz[0, index[[0, 5]] * 3 + 1]
    gt_z_02 = gt_xyz[0, index[[0, 5]] * 3 + 2]
    gt_x_02 = gt_x_02.cpu().numpy()
    gt_y_02 = gt_y_02.cpu().numpy()
    gt_z_02 = gt_z_02.cpu().numpy()
    gt_x_03 = gt_xyz[0, index[[0, 9]] * 3]
    gt_y_03 = gt_xyz[0, index[[0, 9]] * 3 + 1]
    gt_z_03 = gt_xyz[0, index[[0, 9]] * 3 + 2]
    gt_x_03 = gt_x_03.cpu().numpy()
    gt_y_03 = gt_y_03.cpu().numpy()
    gt_z_03 = gt_z_03.cpu().numpy()
    gt_x_04 = gt_xyz[0, index[[0, 13]] * 3]
    gt_y_04 = gt_xyz[0, index[[0, 13]] * 3 + 1]
    gt_z_04 = gt_xyz[0, index[[0, 13]] * 3 + 2]
    gt_x_04 = gt_x_04.cpu().numpy()
    gt_y_04 = gt_y_04.cpu().numpy()
    gt_z_04 = gt_z_04.cpu().numpy()
    gt_x_05 = gt_xyz[0, index[[0, 17]] * 3]
    gt_y_05 = gt_xyz[0, index[[0, 17]] * 3 + 1]
    gt_z_05 = gt_xyz[0, index[[0, 17]] * 3 + 2]
    gt_x_05 = gt_x_05.cpu().numpy()
    gt_y_05 = gt_y_05.cpu().numpy()
    gt_z_05 = gt_z_05.cpu().numpy()
    gt_x_1 = gt_xyz[0, index[1:5] * 3]
    gt_y_1 = gt_xyz[0, index[1:5] * 3 + 1]
    gt_z_1 = gt_xyz[0, index[1:5] * 3 + 2]
    gt_x_1 = gt_x_1.cpu().numpy()
    gt_y_1 = gt_y_1.cpu().numpy()
    gt_z_1 = gt_z_1.cpu().numpy()
    gt_x_2 = gt_xyz[0, index[5:9] * 3]
    gt_y_2 = gt_xyz[0, index[5:9] * 3 + 1]
    gt_z_2 = gt_xyz[0, index[5:9] * 3 + 2]
    gt_x_2 = gt_x_2.cpu().numpy()
    gt_y_2 = gt_y_2.cpu().numpy()
    gt_z_2 = gt_z_2.cpu().numpy()
    gt_x_3 = gt_xyz[0, index[9:13] * 3]
    gt_y_3 = gt_xyz[0, index[9:13] * 3 + 1]
    gt_z_3 = gt_xyz[0, index[9:13] * 3 + 2]
    gt_x_3 = gt_x_3.cpu().numpy()
    gt_y_3 = gt_y_3.cpu().numpy()
    gt_z_3 = gt_z_3.cpu().numpy()
    gt_x_4 = gt_xyz[0, index[13:17] * 3]
    gt_y_4 = gt_xyz[0, index[13:17] * 3 + 1]
    gt_z_4 = gt_xyz[0, index[13:17] * 3 + 2]
    gt_x_4 = gt_x_4.cpu().numpy()
    gt_y_4 = gt_y_4.cpu().numpy()
    gt_z_4 = gt_z_4.cpu().numpy()
    gt_x_5 = gt_xyz[0, index[17:21] * 3]
    gt_y_5 = gt_xyz[0, index[17:21] * 3 + 1]
    gt_z_5 = gt_xyz[0, index[17:21] * 3 + 2]
    gt_x_5 = gt_x_5.cpu().numpy()
    gt_y_5 = gt_y_5.cpu().numpy()
    gt_z_5 = gt_z_5.cpu().numpy()

    outputs_x_01 = outputs_xyz[0, index[[0, 1]] * 3]
    outputs_y_01 = outputs_xyz[0, index[[0, 1]] * 3 + 1]
    outputs_z_01 = outputs_xyz[0, index[[0, 1]] * 3 + 2]
    outputs_x_01 = outputs_x_01.cpu().numpy()
    outputs_y_01 = outputs_y_01.cpu().numpy()
    outputs_z_01 = outputs_z_01.cpu().numpy()
    outputs_x_02 = outputs_xyz[0, index[[0, 5]] * 3]
    outputs_y_02 = outputs_xyz[0, index[[0, 5]] * 3 + 1]
    outputs_z_02 = outputs_xyz[0, index[[0, 5]] * 3 + 2]
    outputs_x_02 = outputs_x_02.cpu().numpy()
    outputs_y_02 = outputs_y_02.cpu().numpy()
    outputs_z_02 = outputs_z_02.cpu().numpy()
    outputs_x_03 = outputs_xyz[0, index[[0, 9]] * 3]
    outputs_y_03 = outputs_xyz[0, index[[0, 9]] * 3 + 1]
    outputs_z_03 = outputs_xyz[0, index[[0, 9]] * 3 + 2]
    outputs_x_03 = outputs_x_03.cpu().numpy()
    outputs_y_03 = outputs_y_03.cpu().numpy()
    outputs_z_03 = outputs_z_03.cpu().numpy()
    outputs_x_04 = outputs_xyz[0, index[[0, 13]] * 3]
    outputs_y_04 = outputs_xyz[0, index[[0, 13]] * 3 + 1]
    outputs_z_04 = outputs_xyz[0, index[[0, 13]] * 3 + 2]
    outputs_x_04 = outputs_x_04.cpu().numpy()
    outputs_y_04 = outputs_y_04.cpu().numpy()
    outputs_z_04 = outputs_z_04.cpu().numpy()
    outputs_x_05 = outputs_xyz[0, index[[0, 17]] * 3]
    outputs_y_05 = outputs_xyz[0, index[[0, 17]] * 3 + 1]
    outputs_z_05 = outputs_xyz[0, index[[0, 17]] * 3 + 2]
    outputs_x_05 = outputs_x_05.cpu().numpy()
    outputs_y_05 = outputs_y_05.cpu().numpy()
    outputs_z_05 = outputs_z_05.cpu().numpy()
    outputs_x_1 = outputs_xyz[0, index[1:5] * 3]
    outputs_y_1 = outputs_xyz[0, index[1:5] * 3 + 1]
    outputs_z_1 = outputs_xyz[0, index[1:5] * 3 + 2]
    outputs_x_1 = outputs_x_1.cpu().numpy()
    outputs_y_1 = outputs_y_1.cpu().numpy()
    outputs_z_1 = outputs_z_1.cpu().numpy()
    outputs_x_2 = outputs_xyz[0, index[5:9] * 3]
    outputs_y_2 = outputs_xyz[0, index[5:9] * 3 + 1]
    outputs_z_2 = outputs_xyz[0, index[5:9] * 3 + 2]
    outputs_x_2 = outputs_x_2.cpu().numpy()
    outputs_y_2 = outputs_y_2.cpu().numpy()
    outputs_z_2 = outputs_z_2.cpu().numpy()
    outputs_x_3 = outputs_xyz[0, index[9:13] * 3]
    outputs_y_3 = outputs_xyz[0, index[9:13] * 3 + 1]
    outputs_z_3 = outputs_xyz[0, index[9:13] * 3 + 2]
    outputs_x_3 = outputs_x_3.cpu().numpy()
    outputs_y_3 = outputs_y_3.cpu().numpy()
    outputs_z_3 = outputs_z_3.cpu().numpy()
    outputs_x_4 = outputs_xyz[0, index[13:17] * 3]
    outputs_y_4 = outputs_xyz[0, index[13:17] * 3 + 1]
    outputs_z_4 = outputs_xyz[0, index[13:17] * 3 + 2]
    outputs_x_4 = outputs_x_4.cpu().numpy()
    outputs_y_4 = outputs_y_4.cpu().numpy()
    outputs_z_4 = outputs_z_4.cpu().numpy()
    outputs_x_5 = outputs_xyz[0, index[17:21] * 3]
    outputs_y_5 = outputs_xyz[0, index[17:21] * 3 + 1]
    outputs_z_5 = outputs_xyz[0, index[17:21] * 3 + 2]
    outputs_x_5 = outputs_x_5.cpu().numpy()
    outputs_y_5 = outputs_y_5.cpu().numpy()
    outputs_z_5 = outputs_z_5.cpu().numpy()

    fig1 = plt.figure()
    plt.imshow(images[0])
    plt.title('depth image')
    fig2 = plt.figure()
    ax = Axes3D(fig2)
    plt.title("point cloud")
    ax.scatter(x, y, z, c="m", marker=".", s=15, linewidths=0, alpha=1, cmap="spectral")
    ax = plt.gca(projection='3d')
    ax.plot(gt_x_01, gt_y_01, gt_z_01, 'go-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_02, gt_y_02, gt_z_02, 'bo-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_03, gt_y_03, gt_z_03, 'ko-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_04, gt_y_04, gt_z_04, 'ro-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_05, gt_y_05, gt_z_05, 'yo-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_1, gt_y_1, gt_z_1, 'go-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_2, gt_y_2, gt_z_2, 'bo-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_3, gt_y_3, gt_z_3, 'ko-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_4, gt_y_4, gt_z_4, 'ro-', linewidth=1, alpha=0.6)
    ax.plot(gt_x_5, gt_y_5, gt_z_5, 'yo-', linewidth=1, alpha=0.6)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.title('GT')
    fig3 = plt.figure()
    ax = Axes3D(fig3)
    plt.title("point cloud")
    ax.scatter(x, y, z, c="m", marker=".", s=15, linewidths=0, alpha=1, cmap="spectral")
    ax = plt.gca(projection='3d')
    ax.plot(outputs_x_01, outputs_y_01, outputs_z_01, 'go-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_02, outputs_y_02, outputs_z_02, 'bo-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_03, outputs_y_03, outputs_z_03, 'ko-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_04, outputs_y_04, outputs_z_04, 'ro-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_05, outputs_y_05, outputs_z_05, 'yo-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_1, outputs_y_1, outputs_z_1, 'go-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_2, outputs_y_2, outputs_z_2, 'bo-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_3, outputs_y_3, outputs_z_3, 'ko-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_4, outputs_y_4, outputs_z_4, 'ro-', linewidth=1, alpha=0.6)
    ax.plot(outputs_x_5, outputs_y_5, outputs_z_5, 'yo-', linewidth=1, alpha=0.6)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.title('Prediction')
    plt.show()