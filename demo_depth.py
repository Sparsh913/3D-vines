import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from demo import get_edge_map
import open3d as o3d

#defining file path
path = '/home/frc-ag-1/Desktop/git/vine_pruning/images/2022-11-09-17-48-16.bag/'

#reading images
imgL = cv2.imread(path+'cam0/0_rectified.png')
imgR = cv2.imread(path+'cam1/0_rectified.png')
grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

heightL, widthL, channelsL = imgL.shape
heightR, widthR, channelsR = imgR.shape

#reading camera matrices npy files
cameraMatrixL = np.load(path+'cam0/K0.npy')
distL = np.load(path+'cam0/D0.npy')
RotL = np.load(path+'cam0/R0.npy')

cameraMatrixR = np.load(path+'cam1/K1.npy')
distR = np.load(path+'cam1/D1.npy')
RotR = np.load(path+'cam1/R1.npy')

projMatrixL = np.load(path+'cam0/P0.npy')
projMatrixR = np.load(path+'cam1/P1.npy')
rectL = projMatrixL[:,:3]
rectR = projMatrixR[:,:3]

newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))


R = RotR @ np.linalg.inv(RotL)
print("Rotation Matrix Relative", R)
T = np.array([-165.1763711199545, 0.0, 0.0])
print("Translation vector Relative", T)

rectifyScale = 1

rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv2.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], R, T, rectifyScale, (0,0))
print("Perspective Transformation Matrix", Q)

#loading disparity map
DISP_DIR = './demo_output/'
disparity = np.load(DISP_DIR  + '0_rectified.npy')
# disparity = np.load('/home/frc-ag-1/Desktop/git/CREStereo-Pytorch/test.npy')

print('disparity map shape: ', disparity.shape)

baseline = np.abs(projMatrixR[0][3] / projMatrixR[0][0])
f_norm = np.abs(cameraMatrixL[0][0])
cx = cameraMatrixL[0][2]
cy = cameraMatrixL[1][2]

IMSHAPE = (1536,2048)
ALL_AX0, ALL_AX1 = np.where(np.ones(IMSHAPE))
ALL_AX0 = ALL_AX0.reshape(IMSHAPE)
ALL_AX1 = ALL_AX1.reshape(IMSHAPE)

stub = -baseline / disparity
xyz = np.dstack((
    stub * (ALL_AX1 - cx),  # columns
    stub * (ALL_AX0 - cy),  # rows
    stub * f_norm,
))

# get edge map 
edge_map = get_edge_map(imgL)

xyz = np.nan_to_num(xyz, copy=False, nan=1e3)
xyz = xyz.reshape(-1, 3)



# remove points with zero in edge map
xyz = xyz[edge_map.reshape(-1) != 0]
#convert bggr to rgb
imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
colors = imgL.reshape(-1, 3)[edge_map.reshape(-1) != 0]/255

# remove points with DEPTH_LIMIT > 2
DEPTH_LIMIT = 1400
xyz1 = xyz[xyz[:, 2] < DEPTH_LIMIT]
colors = colors[xyz[:, 2] < DEPTH_LIMIT]

# visualize point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz1)
#add color to point cloud
pcd.colors = o3d.utility.Vector3dVector(colors)
#visualize point cloud
o3d.visualization.draw_geometries([pcd])

#save ply 

