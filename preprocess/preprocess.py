import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
import open3D as o3d
from farthest_point_sampling_fast import farthest_point_sampling_fast

dataset_dir = '../data/cvpr15_MSRAHandGestureDB/'
save_dir = './'
subject_names = ['HandPointNet']
# gesture_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y']
gesture_names = ['test']

JOINT_NUM = 21
SAMPLE_NUM = 1024
sample_num_level1 = 512
sample_num_level2 = 128

img_width = 424
img_height = 512

for sub_idx in range(len(subject_names)):
    os.mkdir(save_dir + subject_names[sub_idx])
    for ges_idx in range(len(gesture_names)):
        gesture_dir = dataset_dir + subject_names[sub_idx] + '/' + gesture_names[ges_idx]
        typ_data = np.dtype((np.uint8, (img_width, img_height, 3)))
        typ_map = np.dtype((np.float_, (img_width, img_height, 3)))
        Depth_data = np.fromfile(gesture_dir + '/Depdata.txt', dtype=typ_data)
        Depth2xyz = np.fromfile(gesture_dir + '/Depth2xyz.txt', dtype=typ_map)
        frame_num = len(Depth_data)  # the number of images in one hand pose folder
        valid_pixel_num = img_width * img_height

        # 2. get point cloud and surface normal
        save_gesture_dir = save_dir + subject_names[sub_idx] + '/' + gesture_names[ges_idx]
        os.mkdir(save_gesture_dir)

        print(save_gesture_dir)

        Point_Cloud_FPS = np.zeros((frame_num, SAMPLE_NUM, 6))
        Volume_rotate = np.zeros((frame_num, 3, 3))
        Volume_length = np.zeros((frame_num, 1))
        Volume_offset = np.zeros((frame_num, 3))

        for frm_idx in range(frame_num):
            # 2.2 convert depth to xyz
            hand_3d = np.zeros((valid_pixel_num, 3))
            count = 0  # indicate the number of the valid pixels in one frame
            for ii in range(img_height):
                for jj in range(img_width):
                    if (np.any(Depth_data[frm_idx, ii, jj, :] <= 0.0)) | (np.any(Depth2xyz[frm_idx, ii, jj, :] == 0.0)):
                        continue
                    else:
                        count = count + 1
                        hand_3d[count, 0] = Depth2xyz[frm_idx, ii, jj, 0]
                        hand_3d[count, 1] = Depth2xyz[frm_idx, ii, jj, 1]
                        hand_3d[count, 2] = Depth2xyz[frm_idx, ii, jj, 2]
            hand_points = hand_3d[:(count+1), :]

            # 2.3 create OBB
            pca = PCA()
            coeff = pca.fit_transform(hand_points)
            if coeff[1, 0] < 0:
                coeff[:, 0] = -coeff[:, 0]
            if coeff[2, 2] < 0:
                coeff[:, 2] = -coeff[:, 2]
            coeff[:, 1] = np.cross(coeff[:, 2], coeff[:, 0])

            ptCloud = o3d.geometry.PointCloud()
            ptCloud.points = o3d.utility.Vector3dVector(hand_points)

            hand_points_rotate = hand_points * coeff

            # 2.4 sampling
            if len(hand_points) < SAMPLE_NUM:
                tmp = SAMPLE_NUM // len(hand_points)
                rand_ind = []
                for tmp_i in range(tmp):
                    rand_ind = rand_ind.append(list(range(len(hand_points))))
                rand_ind = rand_ind.append(np.random.permutation(len(hand_points))[SAMPLE_NUM % len(hand_points)])
            else:
                rand_ind = np.random.permutation(len(hand_points))[SAMPLE_NUM % len(hand_points)]
            hand_points_sampled = hand_points[rand_ind,:]
            hand_points_rotate_sampled = hand_points_rotate[rand_ind,:]

            # 2.5 compute surface normal
            normal_k = 30
            ne = ptCloud.make_NormalEstimation()
            tree = ptCloud.make_kdtree()
            ne.set_SearchMethod(tree)
            ne.set_KSearch(normal_k)
            normals = ne.compute()
            normals_sampled = normals[rand_ind, :]

            sensorCenter = np.array([0, 0, 0])
            for k in range(SAMPLE_NUM):
                p1 = sensorCenter - hand_points_sampled[k, :]
                # Flip the normal vector if it is not pointing towards the sensor.
                angle = np.arctan2(np.linalg.norm(np.cross(p1, normals_sampled[k, :])), p1 * normals_sampled[k, :].T.conjugate())
                if (angle > (np.pi / 2)) | (angle < (-np.pi / 2)):
                    normals_sampled[k, :] = -normals_sampled[k, :]
            normals_sampled_rotate = normals_sampled * coeff

            # 2.6 Normalize Point Cloud
            x_min_max = [min(hand_points_rotate[:, 1]), max(hand_points_rotate[:, 1])]
            y_min_max = [min(hand_points_rotate[:, 2]), max(hand_points_rotate[:, 2])]
            z_min_max = [min(hand_points_rotate[:, 3]), max(hand_points_rotate[:, 3])]

            scale = 1.2
            bb3d_x_len = scale * (x_min_max[2] - x_min_max[1])
            bb3d_y_len = scale * (y_min_max[2] - y_min_max[1])
            bb3d_z_len = scale * (z_min_max[2] - z_min_max[1])
            max_bb3d_len = bb3d_x_len

            hand_points_normalized_sampled = hand_points_rotate_sampled / max_bb3d_len
            if len(hand_points) < SAMPLE_NUM:
                offset = np.mean(hand_points_rotate) / max_bb3d_len
            else:
                offset = np.mean(hand_points_normalized_sampled)
            hand_points_normalized_sampled = hand_points_normalized_sampled - np.tile(offset, (SAMPLE_NUM, 1))

            # 2.7 FPS Sampling
            pc = np.concatenate((hand_points_normalized_sampled, normals_sampled_rotate), axis=1)
            # 1st level
            sampled_idx_l1 = farthest_point_sampling_fast(hand_points_normalized_sampled, sample_num_level1).T
            other_idx = np.setdiff1d(np.arange(SAMPLE_NUM), sampled_idx_l1)
            new_idx = np.concatenate((sampled_idx_l1, other_idx), axis=0)
            pc = pc[new_idx, :]
            # 2 nd level
            sampled_idx_l2 = farthest_point_sampling_fast(pc[: sample_num_level1, 0: 2], sample_num_level2).T
            other_idx = np.setdiff1d(np.arange(sample_num_level1), sampled_idx_l2)
            new_idx = np.concatenate((sampled_idx_l2, other_idx), axis=0)
            pc[: sample_num_level1, :] = pc[new_idx, :]

            Point_Cloud_FPS[frm_idx, :, :] = pc
            Volume_rotate[frm_idx, :, :] = coeff
            Volume_length[frm_idx] = max_bb3d_len
            Volume_offset[frm_idx, :] = offset

        # 3. save files
        np.save(save_gesture_dir + '/Point_Cloud_FPS.npy', Point_Cloud_FPS)
        np.save(save_gesture_dir + '/Volume_rotate.npy', Volume_rotate)
        np.save(save_gesture_dir + '/Volume_length.npy', Volume_length)
        np.save(save_gesture_dir + '/Volume_offset.npy', Volume_offset)