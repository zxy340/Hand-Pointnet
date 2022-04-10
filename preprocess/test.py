import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

path = '/home/xiaoyu/MyProject/Hand-Pointnet/data/cvpr15_MSRAHandGestureDB/P0/1/000000_depth.jpg'
image = cv2.imread(path)
print(np.shape(image))
cv2.imshow('image', image)
cv2.waitKey(0)