import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from demo import get_edge_map
import open3d as o3d
import copy

# 1) run it on the reisling 
# 2) Clean the code and document 
# 3) Skeletonization and (urdf creation if possible)
# 4) Figure out how to compare the disparities without the ground truth
    # connecteed components after voxelizing. 
# documentation + metrics 

#defining file path
path = '/home/frc-ag-1/Desktop/git/vine_pruning/images/2022-11-10-10-57-11.bag/' 
# path = '/home/frc-ag-1/Desktop/git/vine_pruning/images/2022-11-10-11-28-17.bag/'
# DISP_DIR = './demo_output/'
DISP_DIR = "/home/frc-ag-1/Desktop/git/IGEV/IGEV-Stereo/2022-11-10-10-57-11.bag/"
# DISP_DIR = "../CREStereo-Pytorch/2022-11-10-10-58-26.bag/"
# DISP_DIR = "./outputs/2022-11-10-10-58-26.bag/"


#reading images
imgL_list = []
imgR_list = []
grayL_list = []
grayR_list = []

for i in range(7):
    imgL = cv2.imread(path+'cam0/'+ str(i) +'_rectified.png')
    imgR = cv2.imread(path+'cam1/'+ str(i)+'_rectified.png')
    grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

    imgL_list.append(imgL)
    imgR_list.append(imgR)
    grayL_list.append(grayL)
    grayR_list.append(grayR)

heightL, widthL, channelsL = imgL_list[0].shape
heightR, widthR, channelsR = imgR_list[0].shape

#reading camera matrices npy files
cameraMatrixL_list = []
distL_list = []
RotL_list = []

points_clouds = []
color_clouds = []
for i in range(7):
    cameraMatrixL = np.load(path+'cam0/K'+str(i)+'.npy')
    RotL = np.load(path+'cam0/R'+str(i)+'.npy')

    cameraMatrixR = np.load(path+'cam1/K'+str(i)+'.npy')
    distR = np.load(path+'cam1/D'+str(i)+'.npy')
    RotR = np.load(path+'cam1/R'+str(i)+'.npy')

    projMatrixL = np.load(path+'cam0/P'+str(i)+'.npy')
    projMatrixR = np.load(path+'cam1/P'+str(i)+'.npy')
    rectL = projMatrixL[:,:3]
    rectR = projMatrixR[:,:3]

    #loading disparity map
    disparity = np.load(DISP_DIR  + str(i) +'_rectified.png.npy')
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

    

    xyz = np.nan_to_num(xyz, copy=False, nan=1e3)
    xyz = xyz.reshape(-1, 3)
    imgL_list[i] = cv2.cvtColor(imgL_list[i], cv2.COLOR_BGR2RGB)
    colors = imgL_list[i].reshape(-1, 3)/255

    # get edge map 
    edge_map = get_edge_map(imgL_list[i])
    
    # remove points with zero in edge map
    # xyz = xyz[edge_map.reshape(-1) != 0]
    # colors = colors[edge_map.reshape(-1) != 0]

    #filtering out points that don't have a wooden like color
    hsv = cv2.cvtColor(imgL_list[i], cv2.COLOR_RGB2HSV)
    lower_wooden = [0, 0, 100]
    upper_wooden = [180, 180, 255]
    mask = cv2.inRange(hsv, np.array(lower_wooden), np.array(upper_wooden))
    #combine the mask and edge map usign bitwise and
    mask = np.logical_not(mask)
    mask = np.logical_and(mask, edge_map)


    xyz = xyz[mask.reshape(-1) != 0]
    colors = colors[mask.reshape(-1) != 0]


    

    # remove points with DEPTH_LIMIT > 2
    xyz1 = xyz
    DEPTH_LIMIT = 0.5
    xyz1 = xyz[xyz[:, 2] < DEPTH_LIMIT]
    colors = colors[xyz[:, 2] < DEPTH_LIMIT]

    #remove points with DEPTH_LIMIT < 0.0
    colors = colors[xyz1[:, 2] > -2]
    xyz1 = xyz1[xyz1[:, 2] > -2]

    #remove points with below some brightness threshold
    #brightness is calculated as the sum of rgb values
    # brightness = np.sum(colors, axis=1)
    # colors = colors[brightness > 0.3]
    # xyz1 = xyz1[brightness > 0.3]

    xyz = xyz1

    #append 
    points_clouds.append(xyz)
    color_clouds.append(colors)

initial_transformation_matrix = np.identity(4)
initial_transformation_matrix[0][3] = -0.22493
initial_transformation_matrix[1][3] = 0.0
transformation_matrix_list = [initial_transformation_matrix]


outlier_nb = 40
outlier_std = 0.5
max = 6
for i in range(1, max+1):
    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(points_clouds[i-1])
    pcd_1.colors = o3d.utility.Vector3dVector(color_clouds[i-1])
    #remove outliers
    pcd_1, ind = pcd_1.remove_statistical_outlier(nb_neighbors=outlier_nb,
                                                    std_ratio=outlier_std)  

    pcd_2 = o3d.geometry.PointCloud()
    pcd_2.points = o3d.utility.Vector3dVector(points_clouds[i])
    pcd_2.colors = o3d.utility.Vector3dVector(color_clouds[i])

    #remove outliers
    pcd_2, ind = pcd_2.remove_statistical_outlier(nb_neighbors=outlier_nb,
                                                    std_ratio=outlier_std)

    target_cloud = copy.deepcopy(pcd_1)
    source_cloud = copy.deepcopy(pcd_2)

    #downsample point cloud
    # target_cloud = target_cloud.voxel_down_sample(voxel_size=0.005)
    # source_cloud = source_cloud.voxel_down_sample(voxel_size=0.005)

    for cloud in [target_cloud, source_cloud]:
            cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30),
            )
            cloud.orient_normals_towards_camera_location()

    #now performing icp registration
    icp = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud, max_correspondence_distance=0.003,
        init=transformation_matrix_list[-1],
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # #performing colored icp
    # icp = o3d.pipelines.registration.registration_colored_icp(
    #     source_cloud, target_cloud, max_correspondence_distance=0.003,
    #     init=transformation_matrix_list[-1],
    #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
    #                                                       relative_rmse=1e-6,
    #                                                       max_iteration=100),
    #     estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP())

    transformation_matrix = icp.transformation
    transformation_matrix_list.append(transformation_matrix)
    print('transformation matrix: ', transformation_matrix)

    # source_cloud.transform(transformation_matrix)

    # stitched_cloud = stitched_cloud + source_cloud

    # pcd_1 = copy.deepcopy(source_cloud)

#now transform the point clouds to stitch them together
pcd_1 = o3d.geometry.PointCloud()
pcd_1.points = o3d.utility.Vector3dVector(points_clouds[max])
pcd_1.colors = o3d.utility.Vector3dVector(color_clouds[max])
pcd_1, ind = pcd_1.remove_statistical_outlier(nb_neighbors=outlier_nb,
                                                    std_ratio=outlier_std)  

stitched_cloud = copy.deepcopy(pcd_1)

for i in range(max, 0, -1):

    transformation_matrix = transformation_matrix_list[i]
    stitched_cloud =  stitched_cloud.transform(transformation_matrix)

    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(points_clouds[i-1])
    pcd_1.colors = o3d.utility.Vector3dVector(color_clouds[i-1])
    pcd_1, ind = pcd_1.remove_statistical_outlier(nb_neighbors=outlier_nb,
                                                    std_ratio=outlier_std)

    source_cloud = copy.deepcopy(pcd_1)
    stitched_cloud = stitched_cloud + source_cloud

# exit()


# visualize point cloud
pcd = stitched_cloud
#visualize point cloud
o3d.visualization.draw_geometries([pcd])

#drawing coordinate frame
# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#         size=0.6, origin=[0, 0, 0])
# o3d.visualization.draw_geometries([pcd, mesh_frame])

#save ply 
# exit()

from pc_skeletor import LBC
lbc = LBC(point_cloud=pcd,
          down_sample=0.005,
          filter_nb_neighbors=80,
          max_attraction = 1024)

lbc.extract_skeleton()
lbc.extract_topology()
lbc.visualize()
lbc.show_graph(lbc.skeleton_graph)
lbc.show_graph(lbc.topology_graph)

lbc.save('./output')
# lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
#             steps=300,
#             output='./output')

# generate new skeletons from existing skeletons