import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import open3d as o3d

data_path = '../data/cvpr15_MSRAHandGestureDB/HandPointNet/test/Depdata.txt'
map_path = '../data/cvpr15_MSRAHandGestureDB/HandPointNet/test/Depth2xyz.txt'

typ_data = np.dtype((np.uint8, (424, 512, 3)))
typ_map = np.dtype((np.float32, (424, 512, 3)))
Depth_data = np.fromfile(data_path, dtype=typ_data)
Depth2xyz = np.fromfile(map_path, dtype=typ_map)
hand_3d = np.zeros((424*512, 3))
count = 0  # indicate the number of the valid pixels in one frame
for ii in range(512):
    for jj in range(424):
        if (np.any(Depth_data[0, jj, ii, :] <= 0.0)) | (np.any(Depth2xyz[0, jj, ii, :] == 0.0)):
            continue
        else:
            hand_3d[count, 0] = Depth2xyz[0, jj, ii, 0]
            hand_3d[count, 1] = Depth2xyz[0, jj, ii, 1]
            hand_3d[count, 2] = Depth2xyz[0, jj, ii, 2]
            count = count + 1
# name = '../data/cvpr15_MSRAHandGestureDB/HandPointNet/test/Depth.jpg'
# cv2.imwrite(name, Depth_data[0])
hand_points = hand_3d[:count, :]
print(np.shape(hand_points))
ptCloud = o3d.geometry.PointCloud()
ptCloud.points = o3d.utility.Vector3dVector(hand_points)
o3d.visualization.draw_geometries([ptCloud], window_name="GT")