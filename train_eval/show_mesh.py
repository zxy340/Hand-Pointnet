import open3d as o3d
import numpy as np

mesh = np.load('mesh.npy')
mesh = mesh[:, :3]
print(np.shape(mesh))
ptCloud = o3d.geometry.PointCloud()
ptCloud.points = o3d.utility.Vector3dVector(mesh)
o3d.visualization.draw_geometries([ptCloud], window_name="GT")